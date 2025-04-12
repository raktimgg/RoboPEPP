import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import kornia

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes

# rendering components
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
    PerspectiveCameras,Textures
)

from os.path import exists


class RobotMeshRenderer():
    """
    Class that render robot mesh with differentiable renderer
    """
    def __init__(self, focal_length, principal_point, image_size, robot, mesh_files, device):
        
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.image_size = image_size
        self.device = device
        self.robot = robot
        self.mesh_files = mesh_files
        self.preload_verts = []
        self.preload_faces = []
        

        # preload the mesh to save loading time
        for m_file in mesh_files:
            assert exists(m_file)
            preload_verts_i, preload_faces_idx_i, _ = load_obj(m_file)
            preload_faces_i = preload_faces_idx_i.verts_idx
            self.preload_verts.append(preload_verts_i.to(device))
            self.preload_faces.append(preload_faces_i.to(device))


        # set up differentiable renderer with given camera parameters
        self.cameras = PerspectiveCameras(focal_length = [focal_length],
                                     principal_point = [principal_point],
                                     device=device, 
                                     in_ndc=False, image_size = [image_size]) #  (height, width) !!!!!
        
        blend_params = BlendParams(sigma=1e-8, gamma=1e-8)
        raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
            faces_per_pixel=100, 
            max_faces_per_bin=100000,  # max_faces_per_bin=1000000,  
        )
        
        # Create a silhouette mesh renderer by composing a rasterizer and a shader. 
        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )
        
        
        # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            max_faces_per_bin=100000, 
        )
        # We can add a point light in front of the object. 
        lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
        self.phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(device=device, cameras=self.cameras, lights=lights)
        )
        
    def get_robot_mesh(self, joint_angle):
        
        R_list, t_list = self.robot.get_joint_RT(joint_angle)
        R_list = R_list[0]
        t_list = t_list[0]
        assert len(self.mesh_files) == R_list.shape[0] and len(self.mesh_files) == t_list.shape[0]

        verts_list = []
        faces_list = []
        verts_rgb_list = []
        verts_count = 0

        # Predefined colors for each link
        predefined_colors = [
            torch.tensor([1.0, 0.0, 0.0]),  # Link 0 - Red
            torch.tensor([0.0, 1.0, 0.0]),  # Link 1 - Green
            torch.tensor([0.0, 0.0, 1.0]),  # Link 2 - Blue
            torch.tensor([1.0, 1.0, 0.0]),  # Link 3 - Yellow
            torch.tensor([0.0, 1.0, 1.0]),  # Link 4 - Cyan
            torch.tensor([1.0, 0.0, 1.0]),  # Link 5 - Magenta
            torch.tensor([0.5, 0.5, 0.5]),  # Link 6 - Gray
            torch.tensor([0.7, 0.3, 0.0]),  # Link 7 - Brown
            torch.tensor([0.3, 0.7, 0.7]),  # Hand - Teal
        ]


        for i in range(len(self.mesh_files)):
            verts_i = self.preload_verts[i]
            faces_i = self.preload_faces[i]

            R = R_list[i]
            t = t_list[i]
            verts_i = verts_i @ R.T + t
            #verts_i = (R @ verts_i.T).T + t
            faces_i = faces_i + verts_count

            verts_count+=verts_i.shape[0]

            verts_list.append(verts_i.to(self.device))
            faces_list.append(faces_i.to(self.device))

            # Initialize each vertex to be white in color.
            # color = torch.rand(3).to(self.device)
            color = predefined_colors[i % len(predefined_colors)].to(self.device)
            verts_rgb_i = torch.ones_like(verts_i) * color  # (V, 3)
            verts_rgb_list.append(verts_rgb_i.to(self.device))



        verts = torch.concat(verts_list, dim=0)
        faces = torch.concat(faces_list, dim=0)

        verts_rgb = torch.concat(verts_rgb_list,dim=0)[None]
        textures = Textures(verts_rgb=verts_rgb)

        # Create a Meshes object
        robot_mesh = Meshes(
            verts=[verts.to(self.device)],   
            faces=[faces.to(self.device)], 
            textures=textures
        )
        
        return robot_mesh


    def get_robot_verts_and_faces(self, joint_angle):
        
        R_list, t_list = self.robot.get_joint_RT(joint_angle)
        assert len(self.mesh_files) == R_list.shape[0] and len(self.mesh_files) == t_list.shape[0]

        verts_list = []
        faces_list = []
        verts_rgb_list = []
        verts_count = 0
        for i in range(len(self.mesh_files)):
            verts_i = self.preload_verts[i]
            faces_i = self.preload_faces[i]

            R = torch.tensor(R_list[i],dtype=torch.float32)
            t = torch.tensor(t_list[i],dtype=torch.float32)
            verts_i = verts_i @ R.T + t
            #verts_i = (R @ verts_i.T).T + t
            faces_i = faces_i + verts_count

            verts_count+=verts_i.shape[0]

            verts_list.append(verts_i.to(self.device))
            faces_list.append(faces_i.to(self.device))

            # Initialize each vertex to be white in color.
            #color = torch.rand(3)
            #verts_rgb_i = torch.ones_like(verts_i) * color  # (V, 3)
            #verts_rgb_list.append(verts_rgb_i.to(self.device))

        verts = torch.concat(verts_list, dim=0)
        faces = torch.concat(faces_list, dim=0)

        
        return verts, faces
    

    
def setup_robot_renderer(mesh_files, fx, fy, px, py, height, width, robot):
    # mesh_files: list of mesh files
    focal_length = [-fx,-fy]
    principal_point = [px, py]
    image_size = [height,width]

    robot_renderer = RobotMeshRenderer(
        focal_length=focal_length, principal_point=principal_point, image_size=image_size, 
        robot=robot, mesh_files=mesh_files, device='cuda')

    return robot_renderer
    
def render_single_robot_mask(cTr, robot_mesh, robot_renderer, axis_angle=True):
    # cTr: (6)
    # img: (1, H, W)
    cTr = cTr.float()
    if axis_angle:
        R = kornia.geometry.conversions.axis_angle_to_rotation_matrix(cTr[:3][None]).float()  # (1, 3, 3)
        R = torch.transpose(R,1,2).float()
        T = cTr[3:][None].float()   # (1, 3)
    else:
        R = cTr[:3,:3].unsqueeze(0).float()  # (1, 3, 3)
        T = cTr[:3,3].unsqueeze(0).float()
    #R = to_valid_R_batch(R)
    

    if T[0,-1] < 0:
        rendered_image = robot_renderer.silhouette_renderer(meshes_world=robot_mesh, R = -R, T = -T)
    else:
        rendered_image = robot_renderer.silhouette_renderer(meshes_world=robot_mesh, R = R, T = T)

    if torch.isnan(rendered_image).any():
        rendered_image = torch.nan_to_num(rendered_image)
    return rendered_image[..., 3]

def render_single_robot_color(cTr, robot_mesh, robot_renderer, axis_angle=True):
    """
    Render the robot in color using the Phong renderer.
    """
    cTr = cTr.float()
    if axis_angle:
        # Convert axis-angle rotation to rotation matrix
        R = kornia.geometry.conversions.axis_angle_to_rotation_matrix(cTr[:3][None]).float()  # (1, 3, 3)
        R = torch.transpose(R, 1, 2).float()
        T = cTr[3:][None].float()   # (1, 3)
    else:
        R = cTr[:3, :3].unsqueeze(0).float()  # (1, 3, 3)
        T = cTr[:3, 3].unsqueeze(0).float()
        R = torch.transpose(R, 1, 2).float()

    if T[0, -1] < 0:
        rendered_image = robot_renderer.phong_renderer(meshes_world=robot_mesh, R=-R, T=-T)
    else:
        rendered_image = robot_renderer.phong_renderer(meshes_world=robot_mesh, R=R, T=T)

    # Get the rendered RGB image
    # rendered_image = rendered_image[0, ..., :3].cpu().numpy()  # RGB channels only
    return rendered_image