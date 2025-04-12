import os
import sys

import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')
from datasets.BPnP import BPnP_m3d, batch_transform_3d
import torch 
import torch.nn as nn 
from .backbones import vit
from torch.nn import functional as F 
from datasets.image_proc import batch_transform_points, get_keypoints_from_beliefmap, get_soft_keypoints_from_beliefmap, get_xyz_from_kp
from .robot_arm import BaxterArmPytorch, KukaArmPytorch, PandaArm, PandaArmPytorch

class JointNet(nn.Module):

    def __init__(self, feature_channel, npose):
        
        super(JointNet, self).__init__()

        self.feature_channel = feature_channel
        last_channel = self.feature_channel
        self.n_iter = 4
        self.fc_pose_1 = nn.Linear(self.feature_channel + npose, 1024)
        self.fc_pose_2 = nn.Linear(1024, 1024)
        self.decpose = nn.Linear(1024, npose)
        self.drop1 = nn.Dropout(p=0.3)
        self.drop2 = nn.Dropout(p=0.3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)

        self.init_pose = torch.zeros(npose).unsqueeze(0).float().to('cuda')


    def forward(self, xf):

        batch_size = xf.shape[0]
        init_pose = self.init_pose.expand(batch_size, -1)

        pred_pose = init_pose

        for i in range(self.n_iter):
            xc = torch.cat([xf, pred_pose],1).to(torch.float)
            xc = self.fc_pose_1(xc)
            xc = self.drop1(xc)
            xc = self.fc_pose_2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose

        pred_joints = pred_pose

        return pred_joints

class SpatialSoftmax(nn.Module):
    def __init__(self, temperature=0.1):
        super(SpatialSoftmax, self).__init__()
        self.temperature = temperature

    def forward(self, x):
        h, w = x.size()[-2:]
        # Reshape input to (batch_size, num_keypoints, height * width)
        x = x.view(x.size(0), x.size(1), -1)
        # Apply softmax on the last dimension (spatial locations)
        softmax = torch.nn.functional.softmax(x/self.temperature, dim=-1)
        # Reshape back to (batch_size, num_keypoints, height, width)
        return softmax.view(x.size(0), x.size(1), h, w)

class KeypointNet(nn.Module):
    def __init__(self, in_channels, num_keypoints, dropout_prob=0.4):
        super().__init__()

        # First ConvTranspose2d layer (upsample to 28x28)
        self.deconv1 = nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout_prob)

        # Second ConvTranspose2d layer (upsample to 56x56)
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=dropout_prob)

        # Third ConvTranspose2d layer (upsample to 112x112)
        self.deconv3 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(p=dropout_prob)

        # Fourth ConvTranspose2d layer (upsample to 224x224)
        self.deconv4 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout4 = nn.Dropout(p=dropout_prob)

        self.out_layer1 = nn.Conv2d(256, num_keypoints, kernel_size=1, stride=1)
        # self.out_layer2 = nn.Conv2d(num_keypoints, num_keypoints, kernel_size=1, stride=1)

        # Initialize the weights
        self._initialize_weights()

        self.up_scale = 8

        self.spatial_softmax = SpatialSoftmax(temperature=0.1)

    def _initialize_weights(self):
        # Initialize weights as in the original code
        nn.init.kaiming_normal_(self.deconv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.deconv1.bias, 0)
        
        nn.init.kaiming_normal_(self.deconv2.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.deconv2.bias, 0)
        
        nn.init.kaiming_normal_(self.deconv3.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.deconv3.bias, 0)
        
        nn.init.kaiming_normal_(self.deconv4.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.deconv4.bias, 0)

    def forward(self, x):
        # Apply the layers
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.deconv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout4(x)

        x = x.contiguous()
        x = self.out_layer1(x)

        return torch.sigmoid(x)


class RobotPoseNet(nn.Module):
    def __init__(self, backbone, input_shape=(256,256), patch_size=16, 
                 pred_emb_dim=384, pred_depth=12, npose=6, nkeypoints=7, **kwargs):
        super(RobotPoseNet, self).__init__()

        urdf_file = kwargs["urdf_file"]
        if nkeypoints == 7:
            self.robot = PandaArmPytorch(urdf_file)
        elif nkeypoints == 8:
            self.robot = KukaArmPytorch(urdf_file)
        elif nkeypoints == 17:
            self.robot = BaxterArmPytorch(urdf_file)
        self.backbone_name = backbone
        self.context_backbone = vit.__dict__[backbone](
                        img_size=[input_shape[0]],
                        patch_size=patch_size)
        
        self.predictor_backbone = vit.__dict__['vit_predictor'](
                    num_patches=self.context_backbone.patch_embed.num_patches,
                    embed_dim=self.context_backbone.embed_dim,
                    predictor_embed_dim=pred_emb_dim,
                    depth=pred_depth,
                    num_heads=self.context_backbone.num_heads)

        self.feature_channel = self.context_backbone.embed_dim
        self.feat_drop = nn.Dropout(p=0.3)

        self.npose = npose
        self.joint_net = JointNet(feature_channel=self.feature_channel, npose=npose)

        self.nkeypoints = nkeypoints
        self.keypoint_net = KeypointNet(in_channels=self.feature_channel, num_keypoints=nkeypoints)

        self.joint_mean = kwargs["joint_mean"]
        self.joint_std = kwargs["joint_std"]
    
    def forward(self, x, K, metadata, mask_enc=None, mask_pred=None, soft_keypoints=False, use_gt_joints=False, gt_joints=None):
        B = x.shape[0]
        if mask_enc is None:
            img_feat = self.context_backbone(x)
            img_feat_predictor = self.predictor_backbone(img_feat, None, None)
            img_feat = img_feat_predictor
        else:
            img_feat_context = self.context_backbone(x, mask_enc)
            img_feat_predictor = self.predictor_backbone(img_feat_context, mask_enc, mask_pred)
            img_feat = img_feat_predictor


        # Predicting the joint angles
        xf = img_feat.mean(dim=1)
        pred_joints = self.joint_net(xf)

        # Predicting the keypoint heatmaps
        num_patch = int(np.sqrt(img_feat.shape[1]))
        img_feat = img_feat.permute(0,2,1).view(B, self.feature_channel, num_patch, num_patch)
        pred_keypoints = self.keypoint_net(img_feat)

        if use_gt_joints:
            pred_joints_denorm = gt_joints*self.joint_std + self.joint_mean
        else:
            pred_joints_denorm = pred_joints*self.joint_std + self.joint_mean
        if self.nkeypoints == 7 or self.nkeypoints == 8: # panda or kuka
            pred_joints_denorm = torch.cat([pred_joints_denorm, torch.zeros(B, 1).to(pred_joints_denorm.device)], dim=1)
            _, kp_xyz_rob = self.robot.get_joint_RT(pred_joints_denorm[:,:7])
        elif self.nkeypoints == 17: # baxter
            pred_joints_denorm = torch.cat([pred_joints_denorm, torch.zeros(B, 2).to(pred_joints_denorm.device)], dim=1)
            _, kp_xyz_rob = self.robot.get_joint_RT(pred_joints_denorm)
        if self.nkeypoints == 7: # panda
            kp_xyz_rob = kp_xyz_rob[:, [0,2,3,4,6,7,8], :]

        (pad_w, pad_h) = metadata['pad']
        pad_w, pad_h = pad_w.to(x.device), pad_h.to(x.device)
        (scale_x, scale_y) = metadata['scale']
        scale_x, scale_y = scale_x.to(x.device), scale_y.to(x.device)
        bbox_min = metadata['bbox_min'].to(x.device)

        if not soft_keypoints:
            pred_keypoints_uv = get_keypoints_from_beliefmap(pred_keypoints).float()
        else:
            pred_keypoints_uv = get_soft_keypoints_from_beliefmap(pred_keypoints).float()

        pred_keypoints_uv[:,:,0] -= pad_w.unsqueeze(1)
        pred_keypoints_uv[:,:,1] -= pad_h.unsqueeze(1)
        pred_keypoints_uv[:,:,0] /= scale_x.unsqueeze(1)
        pred_keypoints_uv[:,:,1] /= scale_y.unsqueeze(1)
        pred_keypoints_uv[:,:,0] += bbox_min[:,0].unsqueeze(1)
        pred_keypoints_uv[:,:,1] += bbox_min[:,1].unsqueeze(1)
        
        P_6d = BPnP_m3d.apply(pred_keypoints_uv.detach(), kp_xyz_rob, K[0].float())
        kp_xyz_cam_transformed = batch_transform_3d(P_6d, kp_xyz_rob)

        return pred_joints, pred_keypoints, kp_xyz_cam_transformed, pred_keypoints_uv, kp_xyz_rob



def make_robotposenet(backbone, input_shape=(256,256), patch_size=16, r_path=None, **kwargs):
    model = RobotPoseNet(backbone, input_shape=input_shape, patch_size=patch_size, **kwargs)
    # model.init_weights()
    if backbone in ["vit_base", "vit_large", "vit_huge"]:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'), weights_only=True)
        pretrained_dict = checkpoint['encoder']
        new_state_dict = {}
        for k, v in pretrained_dict.items():
            new_key = k.replace("module.", "")  # Remove the "module." prefix
            new_state_dict[new_key] = v
        model.context_backbone.load_state_dict(new_state_dict)

        pretrained_dict = checkpoint['predictor']
        new_state_dict = {}
        for k, v in pretrained_dict.items():
            new_key = k.replace("module.", "")  # Remove the "module." prefix
            new_state_dict[new_key] = v
        model.predictor_backbone.load_state_dict(new_state_dict)

        print("Loaded pretrained weights for", backbone)
    return model