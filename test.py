import os

from matplotlib import pyplot as plt
import numpy as np
import torch # type: ignore
import argparse
import pprint
import yaml
from tqdm import tqdm
from PIL import Image
from scipy.spatial.transform import Rotation as R
import kornia as kn
import pickle

from datasets.dream import DREAM
from datasets.dream_ssl import DREAM as DREAM_SSL
from datasets.transforms import make_transforms
from datasets.collator import MaskCollator
from models.model import make_robotposenet
from datasets.image_proc import calculate_pck_batch, get_keypoints_from_beliefmap, calculate_pck, get_pnp_thresh, project_to_image_plane, solve_pnp, solve_pnp_ransac
from accelerate import Accelerator, DistributedDataParallelKwargs # type: ignore
from models.robot_arm import BaxterArmPytorch, KukaArmPytorch, PandaArm, PandaArmPytorch
from datasets.BPnP import BPnP, BPnP_m3d, batch_transform_3d



def main(args):
    device = 'cuda'
    robot_name = args['robot']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    num_workers = args['data']['num_workers']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    backbone = args['model']['backbone']
    image_shape = int(args['model']['image_shape'])
    patch_size = args['model']['patch_size']
    r_path = args['model']['jepa_path']
    urdf_file = args['model']['urdf_file']
    epochs = args['training']['epochs']
    pred_emb_dim = args['model']['pred_emb_dim']
    pred_depth = args['model']['pred_depth']

    batch_size = int(args['data']['batch_size']*2)
    use_accelerate = args['training']['use_accelerate']
    pre_computed_bbox = args['data']['pre_computed_bbox']
    test_seq = args['data']['test_seq']

    use_accelerate = False

    if not pre_computed_bbox: # if using grounding dino in evaluation loop using multiple workers can lead to parallel GPU processes
        batch_size = 1
        num_workers = 0

    if use_accelerate:
        # accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
        accelerator = Accelerator()

    if robot_name == 'panda':
        robot = PandaArmPytorch(urdf_file)
    elif robot_name == 'kuka':
        robot = KukaArmPytorch(urdf_file)
    elif robot_name == 'baxter':
        robot = BaxterArmPytorch(urdf_file)

    mask_collator = MaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=(0.0,0.1),
        enc_mask_scale=(0.85,1.0),
        aspect_ratio=(0.75,1.0),
        nenc=1,
        npred=1,
        allow_overlap=False,
        min_keep=-1)
    
    val_transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=False,
        horizontal_flip=False,
        color_distortion=False,
        color_jitter=False)
    
    # DREAM_SSL is used for both testing and sim2real ssl
    # This is mostly identical to DREAM but supports bounding box creation using grounding dino
    val_dataset = DREAM_SSL(
        root=root_path,
        image_folder=image_folder,
        transform=val_transform,
        train=False,
        index_targets=False,
        crop_size = crop_size,
        test_seq=test_seq,
        pre_computed_bbox=pre_computed_bbox,
        )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=mask_collator,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=False)
    
    if robot_name == 'panda':
        joint_mean = torch.Tensor([-5.21536266e-02, 2.67705064e-01,  6.04037923e-03, -2.00518761e+00,
                                1.49018339e-02,  1.98561130e+00]).to(device)
        joint_std = torch.Tensor([1.02490399e+00, 6.45399420e-01, 5.11071070e-01, 5.08277903e-01,
                                7.68895279e-01, 5.10862160e-01]).to(device)
        
        model = make_robotposenet(backbone, input_shape=(image_shape, image_shape), patch_size=patch_size, 
                            r_path=r_path, pred_emb_dim=pred_emb_dim, pred_depth=pred_depth, 
                            joint_mean=joint_mean, joint_std=joint_std, urdf_file=urdf_file).to(device)
    elif robot_name == 'kuka':
        joint_mean = torch.Tensor([-0.5943, -0.3927,  0.8999, 1.6605, -2.7366, -0.8897]).to(device)
        joint_std = torch.Tensor([0.2124, 0.4082, 0.4124, 0.3553, 0.2518, 0.4111]).to(device)
        model = make_robotposenet(backbone, input_shape=(image_shape, image_shape), patch_size=patch_size, 
                                r_path=r_path, pred_emb_dim=pred_emb_dim, pred_depth=pred_depth, 
                                joint_mean=joint_mean, joint_std=joint_std, npose=6, nkeypoints=8, urdf_file=urdf_file).to(device)
    elif robot_name=="baxter":
        joint_mean = torch.Tensor([0.0, -0.0067,  0.0158, -0.0293, -0.0799, 
                                   -0.0302, -0.0054, 0.3613, 0.3594, 0.0037, 
                                   0.0347, 0.0036, -0.0037, 0.0022, -0.0041]).to(device)
        joint_std = torch.Tensor([0.0, 0.8028, 0.8023, 0.7393, 0.7495, 
                                  0.8095, 0.7940, 0.4631, 0.4619, 0.8072,
                                  0.8138, 0.8080, 0.8080, 0.7982, 0.8075]).to(device)
        joint_mean = joint_mean[1:13]
        joint_std = joint_std[1:13]
        model = make_robotposenet(backbone, input_shape=(image_shape, image_shape), patch_size=patch_size, 
                                r_path=r_path, pred_emb_dim=pred_emb_dim, pred_depth=pred_depth, 
                                joint_mean=joint_mean, joint_std=joint_std, npose=12, nkeypoints=17, urdf_file=urdf_file).to(device)

    if robot_name == 'panda':
        checkpoint = torch.load('checkpoints/robopepp_panda.pt', map_location=torch.device('cpu'))['model_state_dict']
    elif robot_name == 'kuka':
        checkpoint = torch.load('checkpoints/robopepp_kuka.pt', map_location=torch.device('cpu'))
    elif robot_name == 'baxter':
        checkpoint = torch.load('checkpoints/robopepp_baxter.pt', map_location=torch.device('cpu'))['model_state_dict']

    pretrained_dict = checkpoint
    new_state_dict = {}
    for k, v in pretrained_dict.items():
        new_key = k.replace("module.", "")  # Remove the "module." prefix
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    print("Loaded pretrained weights for the model.")
    
    criterion_l1 = torch.nn.L1Loss(reduction='none')
    if use_accelerate:
        model, val_loader = accelerator.prepare(
        model, val_loader)


    model.eval()
    val_loss_joints = np.zeros([1,6])
    if robot_name == 'panda':
        val_loss_keypoints = np.zeros([1,7])
        val_l1loss_3d_keypoints = np.zeros([1,7])
    elif robot_name == 'kuka':
        val_loss_keypoints = np.zeros([1,8])
        val_l1loss_3d_keypoints = np.zeros([1,8])
    elif robot_name=="baxter":
        val_loss_joints = np.zeros([1,12])
        val_loss_keypoints = np.zeros([1,17])
        val_l1loss_3d_keypoints = np.zeros([1,17])
    if use_accelerate:
        if accelerator.is_local_main_process:
            iterator = tqdm(val_loader, dynamic_ncols=True)
        else:
            iterator = val_loader
    else:
        iterator = tqdm(val_loader, dynamic_ncols=True)
    with torch.inference_mode():
        pck_list = []
        dist3d = []
        kk = 1
        data_for_plot = {'img_loc':[], 'pred_joints':[], 'pred_cTr':[]}
        for i, ((x, gt_joints, gt_keypoints, metadata), masks_enc, masks_pred) in enumerate(iterator):
            x, gt_joints, gt_keypoints = x.to(device, non_blocking=True), gt_joints.to(device, non_blocking=True), gt_keypoints.to(device, non_blocking=True)
            masks_enc = [u.to(device, non_blocking=True) for u in masks_enc]
            masks_pred = [u.to(device, non_blocking=True) for u in masks_pred]

            masks_enc = None
            masks_pred = None

            gt_kp_3d_cam = metadata['orig_keypoints_3d'].to(device, non_blocking=True)
            K = metadata['K'].to(device, non_blocking=True)

            pred_joints, pred_keypoints, pred_kp_3d_cam, _, _ = model(x, K, metadata, masks_enc, masks_pred)

            if robot_name == 'panda' or robot_name == 'kuka':
                loss_joints = criterion_l1(pred_joints*joint_std + joint_mean, gt_joints[:,:6])
            elif robot_name == 'baxter':
                loss_joints = criterion_l1(pred_joints*joint_std + joint_mean, gt_joints[:,1:13])
            loss_keypoints = criterion_l1(get_keypoints_from_beliefmap(pred_keypoints), get_keypoints_from_beliefmap(gt_keypoints))
            l1loss_3d_keypoints = criterion_l1(pred_kp_3d_cam, gt_kp_3d_cam)

            val_loss_keypoints += loss_keypoints.mean(dim=(0,2)).detach().cpu().numpy()
            val_l1loss_3d_keypoints += l1loss_3d_keypoints.mean(dim=(0,2)).detach().cpu().numpy()

            val_loss_joints += loss_joints.mean(dim=0).detach().cpu().numpy()

            B, C, _, _ = pred_keypoints.shape
            pred_joints = pred_joints*joint_std + joint_mean

            if robot_name == 'panda' or robot_name == 'kuka': # panda or kuka
                pred_joints = torch.cat([pred_joints, torch.zeros(B, 1).to(pred_joints.device)], dim=1)
                _, keypoints_3d_rob_pred = robot.get_joint_RT(pred_joints[:,:7])
            elif robot_name == 'baxter': # baxter
                pred_joints = torch.cat([pred_joints, torch.zeros(B, 2).to(pred_joints.device)], dim=1)
                _, keypoints_3d_rob_pred = robot.get_joint_RT(pred_joints)

            if robot_name == 'panda':
                keypoints_3d_rob_pred = keypoints_3d_rob_pred[:,[0,2,3,4,6,7,8]]


            gt_keypoints_uv = metadata['orig_keypoints'].to(device)
            
            (pad_w, pad_h) = metadata['pad']
            pad_w, pad_h = pad_w.to(x.device), pad_h.to(x.device)
            (scale_x, scale_y) = metadata['scale']
            scale_x, scale_y = scale_x.to(x.device), scale_y.to(x.device)
            bbox_min = metadata['bbox_min'].to(x.device)
            bbox_max = metadata['bbox_max'].to(x.device)

            pred_keypoints_uv = get_keypoints_from_beliefmap(pred_keypoints)

            pred_keypoints_uv[:,:,0] -= pad_w.unsqueeze(1)
            pred_keypoints_uv[:,:,1] -= pad_h.unsqueeze(1)
            pred_keypoints_uv[:,:,0] /= scale_x.unsqueeze(1)
            pred_keypoints_uv[:,:,1] /= scale_y.unsqueeze(1)
            pred_keypoints_uv[:,:,0] += bbox_min[:,0].unsqueeze(1)
            pred_keypoints_uv[:,:,1] += bbox_min[:,1].unsqueeze(1)


            B, C, _, _ = pred_keypoints.shape
            pred_kp_3d_cam = torch.zeros_like(pred_kp_3d_cam)
            for ii in range(B):
                thresh = get_pnp_thresh(metadata['img_path'][ii])
                valid_indices2 = torch.where(pred_keypoints[ii].view(1, C,-1).max(dim=-1).values>thresh)[1]
                thresh -= 0.025
                while valid_indices2.shape[0] < 4:
                    valid_indices2 = torch.where(pred_keypoints[ii].view(1, C,-1).max(dim=-1).values>thresh)[1]
                    thresh -= 0.025

                P_6d = BPnP_m3d.apply(pred_keypoints_uv[ii,valid_indices2][None, ...], keypoints_3d_rob_pred[ii,valid_indices2][None, ...], K[0])
                pred_kp_3d_cam_item = batch_transform_3d(P_6d, keypoints_3d_rob_pred[ii][None, ...], angle_axis=True)
                pred_kp_3d_cam[ii] = pred_kp_3d_cam_item

            data_for_plot['img_loc'].append(metadata['img_path'][0])
            data_for_plot['pred_joints'].append(pred_joints[0].detach().cpu().numpy())
            R_out = kn.geometry.conversions.axis_angle_to_rotation_matrix(P_6d[:, 0:3].view(1, 3))
            PM = torch.cat((R_out[:,0:3,0:3], P_6d[:, 3:6].view(1, 3, 1)), dim=-1)
            PM = torch.cat((PM, torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=device).view(1, 1, 4)), dim=1)
            data_for_plot['pred_cTr'].append(PM[0].detach().cpu().numpy())

            pred_kp_2d = project_to_image_plane(pred_kp_3d_cam, K[0].float())
            l1loss_3d_keypoints = criterion_l1(pred_kp_3d_cam, gt_kp_3d_cam)
            
            dist = torch.norm(pred_kp_3d_cam - gt_kp_3d_cam, dim=-1).mean(-1)

            dist = dist.view(-1).detach().cpu().numpy().tolist()
            dist3d.extend(dist)

            pck = calculate_pck_batch(pred_keypoints_uv.cpu().numpy(), gt_keypoints_uv.cpu().numpy())
            pck_list.extend(pck)


    # for ADD
    dist3d = np.array(dist3d)
    auc_threshold = 0.1
    delta_threshold = 0.00001
    add_threshold_values = np.arange(0.0, auc_threshold, delta_threshold)
    counts_3d = []
    for value in add_threshold_values:
        under_threshold = (
            np.mean(dist3d <= value)
        )
        counts_3d.append(under_threshold)
    auc_add = np.trapz(counts_3d, dx=delta_threshold) / auc_threshold

    ##########################################################
    ##################### LOGGING ############################
    ##########################################################
    if use_accelerate:
        val_loss_joints = accelerator.gather(torch.tensor(val_loss_joints, device=device))
        val_loss_joints = val_loss_joints.mean(dim=0).cpu().numpy()*(180/np.pi) / len(val_loader)  # Averaging over all GPUs

        val_loss_keypoints = accelerator.gather(torch.tensor(val_loss_keypoints, device=device))
        val_loss_keypoints = val_loss_keypoints.mean(dim=0).cpu().numpy() / len(val_loader)  # Averaging over all GPUs

        if accelerator.is_local_main_process:
            # print(f"Epoch {epoch}")
            # Format each element in the tensor to two decimal places
            val_loss_joints_str = ', '.join([f"{loss:.2f}" for loss in val_loss_joints])
            val_loss_keypoints_str = ', '.join([f"{loss:.2f}" for loss in val_loss_keypoints])

            print(f"Testing Loss Joints: [{val_loss_joints_str}]")
            print(f"Testing Loss Keypoints: [{val_loss_keypoints_str}]")
            
    else:
        val_loss_joints /= len(val_loader)  # Standard averaging for non-accelerate case
        val_loss_joints = torch.Tensor(val_loss_joints).mean(dim=0).cpu().numpy()*(180/np.pi)
        val_loss_keypoints /= len(val_loader)  # Standard averaging for non-accelerate case
        val_loss_keypoints = torch.Tensor(val_loss_keypoints).mean(dim=0).cpu().numpy()
        pck_array = np.mean(np.array(pck_list), axis=0)
        val_l1loss_3d_keypoints /= len(val_loader)  # Standard averaging for non-accelerate case

        # Format each element in the tensor to two decimal places
        val_loss_joints_str = ', '.join([f"{loss:.2f}" for loss in val_loss_joints])
        val_loss_keypoints_str = ', '.join([f"{loss:.2f}" for loss in val_loss_keypoints])
        val_pck_str = ', '.join([f"{loss:.2f}" for loss in pck_array])
        val_l1loss_3d_keypoints_str = ', '.join([f"{loss:.2f}" for loss in val_l1loss_3d_keypoints[0]])

        print(f"Testing Joints Error (L1): [{val_loss_joints_str}]")
        print(f"Testing Keypoints Error (L1): [{val_loss_keypoints_str}]")
        print(f"Testing 3D Keypoints Error (L1): [{val_l1loss_3d_keypoints_str}]")
        print(f"Testing PCK: [{val_pck_str}]")
        print(f"Testing ADD AUC: {(auc_add)*100}")
        print(f"Testing mean distance: {np.mean(dist3d)}")
            


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--config', type=str, default='configs/panda.yaml', help='Path to the config file')
    fname = parser.parse_args().config
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(params)
    main(params)