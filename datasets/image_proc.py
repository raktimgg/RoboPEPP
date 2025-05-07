import random
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation as R
from scipy.ndimage.filters import gaussian_filter
import torch # type: ignore
import torchvision.transforms.functional as TVTransformsFunc # type: ignore
import torch.nn.functional as F

def create_belief_map(
        image_resolution,
        # image size (width x height)
        pointsBelief,
        # list of points to draw in a 7x2 tensor
        sigma=2
        # the size of the point
        # returns a tensor of n_points x h x w with the belief maps
    ):

    # Input argument handling
    assert (
        len(image_resolution) == 2
    ), 'Expected "image_resolution" to have length 2, but it has length {}.'.format(
        len(image_resolution)
    )
    image_width, image_height = image_resolution
    image_transpose_resolution = (image_height, image_width)
    out = np.zeros((len(pointsBelief), image_height, image_width))

    w = int(sigma * 2)

    for i_point, point in enumerate(pointsBelief):
        pixel_u = int(point[0])
        pixel_v = int(point[1])
        array = np.zeros(image_transpose_resolution)

        # TODO makes this dynamics so that 0,0 would generate a belief map.
        if (
            pixel_u - w >= 0
            and pixel_u + w + 1 < image_width
            and pixel_v - w >= 0
            and pixel_v + w + 1 < image_height
        ):
            for i in range(pixel_u - w, pixel_u + w + 1):
                for j in range(pixel_v - w, pixel_v + w + 1):
                    array[j, i] = np.exp(
                        -(
                            ((i - pixel_u) ** 2 + (j - pixel_v) ** 2)
                            / (2 * (sigma ** 2))
                        )
                    )
        out[i_point] = array

    return out

def get_keypoints_from_beliefmap(belief_maps):
    # Assuming belief_maps and gt_keypoints have shapes (B, 7, H, W)
    B, C, H, W = belief_maps.shape

    # Flatten the spatial dimensions (H, W) into a single dimension for argmax
    pred_flat = belief_maps.view(B, C, -1)  # (B, 7, H*W)

    # Get the argmax over the flattened dimension (which gives index in H*W)
    pred_max_idx = pred_flat.argmax(dim=-1)    # (B, 7)

    # Convert the 1D indices into 2D coordinates (row, col)
    pred_y = pred_max_idx // W                 # (B, 7), row coordinate
    pred_x = pred_max_idx % W                  # (B, 7), column coordinate

    # Stack the coordinates to get shape (B, 7, 2) for both predicted and ground truth keypoints
    pred_coords = torch.stack((pred_x, pred_y), dim=-1).float()  # (B, 7, 2)
    return pred_coords

def get_soft_keypoints_from_beliefmap(belief_maps, temperature=0.0001):
    # Assuming belief_maps has shape (B, 7, H, W)
    B, C, H, W = belief_maps.shape

    # Apply softmax to get probabilities for each pixel location
    belief_maps = belief_maps.view(B, C, -1)  # Flatten H*W to prepare for softmax
    softmax_maps = F.softmax(belief_maps / temperature, dim=-1).view(B, C, H, W)  # Reshape back to (B, C, H, W)

    # Create coordinate grids for x and y
    coords_x = torch.linspace(0, W - 1, W, device=belief_maps.device)
    coords_y = torch.linspace(0, H - 1, H, device=belief_maps.device)
    coords_x, coords_y = torch.meshgrid(coords_x, coords_y)
    coords_x = coords_x.t().view(1, 1, H, W)  # Transpose and reshape to (1, 1, H, W)
    coords_y = coords_y.t().view(1, 1, H, W)

    # Use softmax maps as weights to calculate expected x and y coordinates
    pred_x = (softmax_maps * coords_x).sum(dim=(2, 3))  # Weighted sum along H and W
    pred_y = (softmax_maps * coords_y).sum(dim=(2, 3))

    # Stack coordinates to get shape (B, 7, 2)
    pred_coords = torch.stack((pred_x, pred_y), dim=-1)  # Shape: (B, C, 2)
    return pred_coords

def get_xyz_from_kp(kp_uv, kp_z, K):
    # Extract intrinsics from K
    f_x = K[:, 0, 0]  # Focal length in x
    f_y = K[:, 1, 1]  # Focal length in y
    c_x = K[:, 0, 2]  # Principal point in x
    c_y = K[:, 1, 2]  # Principal point in y
    
    # Calculate the 3D coordinates (x, y, z) for each keypoint
    u = kp_uv[..., 0]  # Shape: (B, num_keypoints)
    v = kp_uv[..., 1]  # Shape: (B, num_keypoints)
    z = kp_z  # Shape: (B, num_keypoints)

    # Compute x, y, z in 3D
    x_world = (u - c_x.unsqueeze(1)) * z / f_x.unsqueeze(1)
    y_world = (v - c_y.unsqueeze(1)) * z / f_y.unsqueeze(1)
    z_world = z  # z is already the depth

    # Stack x, y, z to get the 3D coordinates
    kp_xyz = torch.stack((x_world, y_world, z_world), dim=-1)  # Shape: (B, num_keypoints, 3)
    return kp_xyz


def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion to a 3x3 rotation matrix.
    Args:
        q: (B, 4) quaternion in (x, y, z, w) format.
    Returns:
        rot_matrix: (B, 3, 3) batched rotation matrices.
    """
    B = q.shape[0]
    # Normalize the quaternion to avoid errors due to non-unit quaternions
    q = F.normalize(q, p=2, dim=-1)
    
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Rotation matrix from quaternion
    rot_matrix = torch.zeros((B, 3, 3), dtype=q.dtype, device=q.device)
    
    rot_matrix[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    rot_matrix[:, 0, 1] = 2 * (x*y - z*w)
    rot_matrix[:, 0, 2] = 2 * (x*z + y*w)
    
    rot_matrix[:, 1, 0] = 2 * (x*y + z*w)
    rot_matrix[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    rot_matrix[:, 1, 2] = 2 * (y*z - x*w)
    
    rot_matrix[:, 2, 0] = 2 * (x*z - y*w)
    rot_matrix[:, 2, 1] = 2 * (y*z + x*w)
    rot_matrix[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    
    return rot_matrix


def batch_transform_points(points, t, q):
    """
    Transform the points from robot frame to camera frame using batched transformations.
    Args:
        points: (B, 7, 3) points in the robot frame.
        t: (B, 3) translation vectors.
        q: (B, 4) quaternions representing rotation.
    Returns:
        transformed_points: (B, 7, 3) points transformed into the camera frame.
    """
    B = points.shape[0]
    
    # Get the rotation matrix from the quaternion
    rot_matrix = quaternion_to_rotation_matrix(q)  # (B, 3, 3)
    
    # Apply the rotation
    transformed_points = torch.bmm(rot_matrix, points.transpose(1, 2)).transpose(1, 2)  # (B, 7, 3)
    
    # Apply the translation
    transformed_points = transformed_points + t.unsqueeze(1)  # (B, 7, 3)
    
    return transformed_points


# Function to filter keypoints based on whether they are inside the image bounds (camera's frustum)
def filter_keypoints_in_frustum(gt_keypoints, pred_keypoints, image_width=640, image_height=480):
    valid_indices = np.where(
        (gt_keypoints[:, 0] >= 0) & (gt_keypoints[:, 0] <= image_width) &
        (gt_keypoints[:, 1] >= 0) & (gt_keypoints[:, 1] <= image_height)
    )[0]
    return gt_keypoints[valid_indices], pred_keypoints[valid_indices]


# Define a function to calculate PCK at different thresholds
def calculate_pck(pred_keypoints, gt_keypoints, image_width=640, image_height=480, thresholds=[2.5,5.0,10.0]):
    # Calculate Euclidean distances between predicted and ground truth keypoints
    distances = np.linalg.norm(pred_keypoints - gt_keypoints, axis=1)
    distances = distances.reshape(-1, 1)
    valid_indices = np.where(
        (gt_keypoints[:, 0] > 0) & (gt_keypoints[:, 0] < image_width) &
        (gt_keypoints[:, 1] > 0) & (gt_keypoints[:, 1] < image_height)
    )[0]
    distances = distances[valid_indices]
    # Calculate PCK for each threshold
    pck_scores = []
    for threshold in thresholds:
        correct_keypoints = (distances <= threshold).sum()
        pck = (correct_keypoints / len(gt_keypoints)) * 100
        pck_scores.append(pck)

    return pck_scores

def calculate_pck_batch(pred_keypoints, gt_keypoints, image_width=640, image_height=480, thresholds=[2.5, 5.0, 10.0]):
    """
    pred_keypoints: predicted keypoints of shape (B, 7, 2)
    gt_keypoints: ground truth keypoints of shape (B, 7, 2)
    image_width: width of the image
    image_height: height of the image
    thresholds: list of thresholds for PCK calculation
    """
    batch_size = pred_keypoints.shape[0]
    pck_scores = []
    for i in range(batch_size):
        pck_scores.append(calculate_pck(pred_keypoints[i], gt_keypoints[i], image_width, image_height, thresholds))
    return pck_scores

def convert_rvec_to_quaternion(rvec):
    """Convert rvec (which is log quaternion) to quaternion"""
    theta = np.sqrt(
        rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]
    )  # in radians
    raxis = [rvec[0] / theta, rvec[1] / theta, rvec[2] / theta]

    # pyrr's Quaternion (order is XYZW), https://pyrr.readthedocs.io/en/latest/oo_api_quaternion.html
    quaternion = Quaternion.from_axis_rotation(raxis, theta)
    quaternion.normalize()
    return quaternion

def solve_pnp(
    canonical_points,
    projections,
    camera_K,
    method=cv2.SOLVEPNP_EPNP,
    refinement=True,
    dist_coeffs=np.array([]),
):

    n_canonial_points = len(canonical_points)
    n_projections = len(projections)
    assert (
        n_canonial_points == n_projections
    ), "Expected canonical_points and projections to have the same length, but they are length {} and {}.".format(
        n_canonial_points, n_projections
    )

    # Process points to remove any NaNs
    canonical_points_proc = []
    projections_proc = []
    for canon_pt, proj in zip(canonical_points, projections):

        if (
            canon_pt is None
            or len(canon_pt) == 0
            or canon_pt[0] is None
            or canon_pt[1] is None
            or proj is None
            or len(proj) == 0
            or proj[0] is None
            or proj[1] is None
        ):
            continue

        canonical_points_proc.append(canon_pt)
        projections_proc.append(proj)

    # Return if no valid points
    if len(canonical_points_proc) == 0:
        return False, None, None

    canonical_points_proc = np.array(canonical_points_proc)
    projections_proc = np.array(projections_proc)

    # Use cv2's PNP solver
    try:
        pnp_retval, rvec, tvec = cv2.solvePnP(
            canonical_points_proc.reshape(canonical_points_proc.shape[0], 1, 3),
            projections_proc.reshape(projections_proc.shape[0], 1, 2),
            camera_K,
            dist_coeffs,
            flags=method,
        )

        if refinement:
            pnp_retval, rvec, tvec = cv2.solvePnP(
                canonical_points_proc.reshape(canonical_points_proc.shape[0], 1, 3),
                projections_proc.reshape(projections_proc.shape[0], 1, 2),
                camera_K,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=True,
                rvec=rvec,
                tvec=tvec,
            )
        translation = tvec[:, 0]
        quaternion = convert_rvec_to_quaternion(rvec[:, 0])

    except:
        pnp_retval = False
        translation = None
        quaternion = None

    return pnp_retval, translation, quaternion

def solve_pnp_ransac(
    canonical_points,
    projections,
    camera_K,
    method=cv2.SOLVEPNP_EPNP,
    refinement=True,
    dist_coeffs=np.array([]),
    inlier_thresh_px=50.0,
    ransac_iterations=1000,
    ransac_confidence=0.99,
):

    n_canonial_points = len(canonical_points)
    n_projections = len(projections)
    
    assert (
        n_canonial_points == n_projections
    ), "Expected canonical_points and projections to have the same length, but they are length {} and {}.".format(
        n_canonial_points, n_projections
    )

    # Process points to remove any NaNs
    canonical_points_proc = []
    projections_proc = []
    for canon_pt, proj in zip(canonical_points, projections):
        if (
            canon_pt is None
            or len(canon_pt) == 0
            or canon_pt[0] is None
            or canon_pt[1] is None
            or proj is None
            or len(proj) == 0
            or proj[0] is None
            or proj[1] is None
        ):
            continue

        canonical_points_proc.append(canon_pt)
        projections_proc.append(proj)

    # Return if no valid points
    if len(canonical_points_proc) == 0:
        return False, None, None

    canonical_points_proc = np.array(canonical_points_proc)
    projections_proc = np.array(projections_proc)

    # Use cv2's PNP solver with RANSAC
    try:
        pnp_retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            canonical_points_proc.reshape(canonical_points_proc.shape[0], 1, 3),
            projections_proc.reshape(projections_proc.shape[0], 1, 2),
            camera_K,
            dist_coeffs,
            useExtrinsicGuess=False,
            iterationsCount=ransac_iterations,
            reprojectionError=inlier_thresh_px,
            confidence=ransac_confidence,
            flags=method,
        )

        if refinement and pnp_retval and len(inliers) > 0:
            # Refine with iterative PnP using inliers
            inlier_points_3d = canonical_points_proc[inliers[:, 0]]
            inlier_projections = projections_proc[inliers[:, 0]]

            pnp_retval, rvec, tvec = cv2.solvePnP(
                inlier_points_3d.reshape(inlier_points_3d.shape[0], 1, 3),
                inlier_projections.reshape(inlier_projections.shape[0], 1, 2),
                camera_K,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=True,
                rvec=rvec,
                tvec=tvec,
            )

        translation = tvec[:, 0] 
        quaternion = convert_rvec_to_quaternion(rvec[:, 0]) 

    except:
        pnp_retval = False
        translation = None
        quaternion = None
        inliers = None
    return pnp_retval, translation, quaternion, inliers



def solve_pnp_ransac_old(
    canonical_points,
    projections,
    camera_K,
    method=cv2.SOLVEPNP_EPNP,
    inlier_thresh_px=40.0,  # this is the threshold for each point to be considered an inlier
    dist_coeffs=np.array([]),
):

    n_canonial_points = len(canonical_points)
    n_projections = len(projections)
    assert (
        n_canonial_points == n_projections
    ), "Expected canonical_points and projections to have the same length, but they are length {} and {}.".format(
        n_canonial_points, n_projections
    )

    # Process points to remove any NaNs
    canonical_points_proc = []
    projections_proc = []
    for canon_pt, proj in zip(canonical_points, projections):

        if (
            canon_pt is None
            or len(canon_pt) == 0
            or canon_pt[0] is None
            or canon_pt[1] is None
            or proj is None
            or len(proj) == 0
            or proj[0] is None
            or proj[1] is None
        ):
            continue

        canonical_points_proc.append(canon_pt)
        projections_proc.append(proj)

    # Return if no valid points
    if len(canonical_points_proc) == 0:
        return False, None, None, None

    canonical_points_proc = np.array(canonical_points_proc)
    projections_proc = np.array(projections_proc)

    # Use cv2's PNP solver
    try:
        pnp_retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            canonical_points_proc.reshape(canonical_points_proc.shape[0], 1, 3),
            projections_proc.reshape(projections_proc.shape[0], 1, 2),
            camera_K,
            dist_coeffs,
            reprojectionError=inlier_thresh_px,
            flags=method,
        )

        translation = tvec[:, 0]
        quaternion = convert_rvec_to_quaternion(rvec[:, 0])

    except:
        pnp_retval = False
        translation = None
        quaternion = None
        inliers = None

    return pnp_retval, translation, quaternion, inliers

def get_bbox(bbox,w,h, strict=True):
    assert len(bbox)==4
    wmin, hmin, wmax, hmax = bbox
    if wmax<0 or hmax <0 or wmin > w or hmin > h:
        print("wmax",wmax,"hmax",hmax,"wmin",wmin,"hmin",hmin)
    wmin,hmin,wmax,hmax=max(0,wmin),max(0,hmin),min(w,wmax),min(h,hmax)
    wnew=wmax-wmin
    hnew=hmax-hmin
    # wmin=int(max(0,wmin-0.3*wnew))
    # wmax=int(min(w,wmax+0.3*wnew))
    # hmin=int(max(0,hmin-0.3*hnew))
    # hmax=int(min(h,hmax+0.3*hnew))
    wnew=wmax-wmin
    hnew=hmax-hmin
    
    if not strict:
        randomw = (random.random()-0.2)/2
        randomh = (random.random()-0.2)/2
        
        dwnew=randomw*wnew
        wmax+=dwnew/2
        wmin-=dwnew/2

        dhnew=randomh*hnew
        hmax+=dhnew/2
        hmin-=dhnew/2
        
        wmin=int(max(0,wmin))
        wmax=int(min(w,wmax))
        hmin=int(max(0,hmin))
        hmax=int(min(h,hmax))
        wnew=wmax-wmin
        hnew=hmax-hmin
    
    # if wnew < 150:
    #     wmax+=75
    #     wmin-=75
    # if hnew < 120:
    #     hmax+=60
    #     hmin-=60
        
    wmin,hmin,wmax,hmax=max(0,wmin),max(0,hmin),min(w,wmax),min(h,hmax)
    wmin,hmin,wmax,hmax=min(w,wmin),min(h,hmin),max(0,wmax),max(0,hmax)
    new_bbox = np.array([wmin,hmin,wmax,hmax])
    return new_bbox


def project_to_image_plane(pred_kp_3d_cam, K):
    """
    Project 3D keypoints from camera frame to image plane using camera intrinsics.

    Args:
    - pred_kp_3d_cam (torch.Tensor): 3D keypoints in camera frame of shape (B, 7, 3).
    - K (torch.Tensor): Camera intrinsics matrix of shape (3, 3).

    Returns:
    - torch.Tensor: Projected 2D points on the image plane of shape (B, 7, 2).
    """
    B, num_kp, _ = pred_kp_3d_cam.shape

    # Reshape for matrix multiplication: (B, 7, 3) -> (B*7, 3)
    points_3d = pred_kp_3d_cam.view(-1, 3).float()
    
    # Project to image plane: (B*7, 3) x (3, 3)^T -> (B*7, 3)
    projected_points_homogeneous = torch.mm(points_3d, K.T)

    # Reshape back to (B, 7, 3)
    projected_points_homogeneous = projected_points_homogeneous.view(B, num_kp, 3)

    # Normalize homogeneous coordinates to get (u', v')
    u = projected_points_homogeneous[:, :, 0] / projected_points_homogeneous[:, :, 2]
    v = projected_points_homogeneous[:, :, 1] / projected_points_homogeneous[:, :, 2]

    # Stack to get final 2D points (B, 7, 2)
    projected_2d_points = torch.stack((u, v), dim=-1)

    return projected_2d_points


def get_extended_bbox(wmin, hmin, wmax, hmax, img_loc):
    if "panda_synth_test_photo" in img_loc:
        return np.array([wmin-10, hmin-10, wmax+10, hmax+10])
    if "panda_synth_test_dr" in img_loc:
        return np.array([wmin-40, hmin-30, wmax+10, hmax+10])
    if "azure" in img_loc:
        return np.array([wmin-10, hmin-30, wmax+20, hmax])
    if "kinect" in img_loc:
        return np.array([wmin-150, hmin-100, wmax+200, hmax+200])
    if "realsense" in img_loc:
        return np.array([wmin-150, hmin, wmax+50, hmax])
    if "orb" in img_loc:
        return np.array([wmin, hmin, wmax, hmax])
    if "kuka_synth_test_photo" in img_loc:
        return np.array([wmin-100, hmin-50, wmax+50, hmax+50])
    if "kuka_synth_test_dr" in img_loc:
        return np.array([wmin-100, hmin-100, wmax+50, hmax+50])
    if "baxter_synth_test_dr" in img_loc:
        return np.array([wmin, hmin, wmax, hmax])
        # return np.array([0, 0, 640, 480])
    if "franka_right" in img_loc:
        return np.array([wmin-100, hmin-100, wmax+100, hmax+100])
    if "franka_left" in img_loc:
        return np.array([wmin-150, hmin-150, wmax+50, hmax+100])
    # wmin, hmin = np.min(kp,axis=0)
    # wmax, hmax = np.max(kp,axis=0)
    # return np.array([wmin-50, hmin-50, wmax+50, hmax+50])
    
def get_pnp_thresh(img_loc, ssl=False):
    if "panda_synth_test_photo" in img_loc:
        return 0.35
    if "panda_synth_test_dr" in img_loc:
        return 0.325
    if ssl:
        if "azure" in img_loc:
            return 0.3
        if "kinect" in img_loc:
            return 0.3
        if "realsense" in img_loc:
            return 0.6
        if "orb" in img_loc:
            return 0.3
    if "azure" in img_loc:
        return 0.5
    if "kinect" in img_loc:
        return 0.6
    if "realsense" in img_loc:
        return 0.8
    if "orb" in img_loc:
        return 0.4
    if "kuka_synth_test_photo" in img_loc:
        return 0.1
    if "kuka_synth_test_dr" in img_loc:
        return 0.1
    if "baxter_synth_test_dr" in img_loc:
        return 0.6
    if "franka_right" in img_loc:
        return 0.6
    if "franka_left" in img_loc:
        return 0.6