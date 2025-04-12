import numpy as np
from roboticstoolbox.robot.ERobot import ERobot

import pytorch_kinematics as pk
import torch

class BaxterLeftArm():
    def __init__(self, urdf_file): 
        
        self.robot = self.Baxter(urdf_file)
        
    def get_joint_RT(self, joint_angle):
        
        assert joint_angle.shape[0] == 7
        joint_angle_all = np.zeros(15)
        joint_angle_all[-7:] = joint_angle

        link_idx_list = [30,31,32,33,34,36,37]
        R_list = []
        t_list = []
        # base:30, J1:30, J2:31, J3:32, J4:33, J5:34, J6:36, J7:37

        for i in range(joint_angle.shape[0]):
            link_idx = link_idx_list[i]
            T = self.robot.fkine(joint_angle_all, end = self.robot.links[link_idx], start = self.robot.links[30])
            R_list.append(T.R)
            t_list.append(T.t)



        return np.array(R_list),np.array(t_list)
        
        
    class Baxter(ERobot):
        """
        Class that imports a URDF model
        """

        def __init__(self, urdf_file):

            links, name, urdf_string, urdf_filepath = self.URDF_read(urdf_file)

            super().__init__(
                links,
                name=name,
                manufacturer="Rethink",
                urdf_string=urdf_string,
                urdf_filepath=urdf_filepath,
                # gripper_links=elinks[9]
            )

            # self.qdlim = np.array([
            #     2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 3.0, 3.0])

    
class PandaArm():
    def __init__(self, urdf_file): 
        
        self.robot = self.Panda(urdf_file)
        
    def get_joint_RT(self, joint_angle):
        
        assert joint_angle.shape[0] == 7


        link_idx_list = [0,1,2,3,4,5,6,7,9]
        # link 0,1,2,3,4,5,6,7, and hand
        R_list = []
        t_list = []
        

        for i in range(len(link_idx_list)):
            link_idx = link_idx_list[i]
            T = self.robot.fkine(joint_angle, end = self.robot.links[link_idx], start = self.robot.links[0])
            R_list.append(T.R)
            t_list.append(T.t)



        return np.array(R_list),np.array(t_list)
        
        
    class Panda(ERobot):
        """
        Class that imports a URDF model
        """

        def __init__(self, urdf_file):
            links, name, urdf_string, urdf_filepath = self.URDF_read(urdf_file)

            super().__init__(
                links,
                name=name,
                manufacturer="Franka",
                urdf_string=urdf_string,
                urdf_filepath=urdf_filepath,
                # gripper_links=elinks[9]
            )

    
class PandaArmPytorch():
    def __init__(self, urdf_file, device='cuda'): 
        self.link_names = ['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 
                           'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7', 'panda_hand']
        self.robot = pk.build_serial_chain_from_urdf(open(urdf_file).read(), self.link_names[-1], self.link_names[0])
        self.robot = self.robot.to(dtype=torch.float32, device=device)

    def get_joint_RT(self, joint_angle):
        R = torch.zeros((joint_angle.shape[0], len(self.link_names), 3, 3)).to(joint_angle.device)
        t = torch.zeros((joint_angle.shape[0], len(self.link_names), 3)).to(joint_angle.device)
        tg_batch = self.robot.forward_kinematics(joint_angle, end_only=False)
        for idx, link_name in enumerate(self.link_names):
            t[:,idx] = tg_batch[link_name].get_matrix()[:,:3,3]
            R[:,idx] = tg_batch[link_name].get_matrix()[:,:3,:3]
        return R, t
    
class KukaArmPytorch():
    def __init__(self, urdf_file): 
        self.link_names = ['iiwa_link_0', 'iiwa_link_1', 'iiwa_link_2', 'iiwa_link_3',
                            'iiwa_link_4', 'iiwa_link_5', 'iiwa_link_6', 'iiwa_link_7']
        with open(urdf_file, 'rb') as f:
            urdf_data = f.read()
        self.robot = pk.build_serial_chain_from_urdf(urdf_data, self.link_names[-1], self.link_names[0])
        self.robot = self.robot.to(dtype=torch.float32, device='cuda')

    def get_joint_RT(self, joint_angle):
        R = torch.zeros((joint_angle.shape[0], len(self.link_names), 3, 3)).to(joint_angle.device)
        t = torch.zeros((joint_angle.shape[0], len(self.link_names), 3)).to(joint_angle.device)
        tg_batch = self.robot.forward_kinematics(joint_angle, end_only=False)
        for idx, link_name in enumerate(self.link_names):
            t[:,idx] = tg_batch[link_name].get_matrix()[:,:3,3]
            R[:,idx] = tg_batch[link_name].get_matrix()[:,:3,:3]
        return R, t
    

class BaxterArmPytorch():
    def __init__(self, urdf_file): 
        KEYPOINT_NAMES_TO_LINK_NAMES = {'torso_t0':'torso', 
                            'right_s0':'right_upper_shoulder', 'left_s0':'left_upper_shoulder',
                            'right_s1':'right_lower_shoulder', 'left_s1':'left_lower_shoulder',
                            'right_e0':'right_upper_elbow','left_e0':'left_upper_elbow', 
                            'right_e1':'right_lower_elbow','left_e1':'left_lower_elbow',
                            'right_w0':'right_upper_forearm', 'left_w0':'left_upper_forearm',
                            'right_w1':'right_lower_forearm', 'left_w1':'left_lower_forearm',
                            'right_w2':'right_wrist', 'left_w2':'left_wrist',
                            'right_hand':'right_hand','left_hand':'left_hand'}
        self.keypoint_names = ['torso_t0', 'right_s0','left_s0', 'right_s1', 'left_s1',
                        'right_e0','left_e0', 'right_e1','left_e1','right_w0', 'left_w0',
                        'right_w1','left_w1','right_w2', 'left_w2','right_hand','left_hand']
        self.left_keypoint_names = ['torso_t0', 'left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2', 'left_hand']
        self.right_keypoint_names = ['torso_t0', 'right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2', 'right_hand']
        
        self.link_names = [KEYPOINT_NAMES_TO_LINK_NAMES[link_name] for link_name in self.keypoint_names]
        self.left_link_names = [KEYPOINT_NAMES_TO_LINK_NAMES[link_name] for link_name in self.left_keypoint_names]
        self.right_link_names = [KEYPOINT_NAMES_TO_LINK_NAMES[link_name] for link_name in self.right_keypoint_names]

        with open(urdf_file, 'rb') as f:
            urdf_data = f.read()

        self.left_robot = pk.build_serial_chain_from_urdf(urdf_data, self.left_link_names[-1], self.left_link_names[0])
        self.left_robot = self.left_robot.to(dtype=torch.float32, device='cuda')

        self.right_robot = pk.build_serial_chain_from_urdf(urdf_data, self.right_link_names[-1], self.right_link_names[0])
        self.right_robot = self.right_robot.to(dtype=torch.float32, device='cuda')

    def get_joint_RT(self, joint_angle):
        R = torch.zeros((joint_angle.shape[0], len(self.link_names), 3, 3)).to(joint_angle.device)
        t = torch.zeros((joint_angle.shape[0], len(self.link_names), 3)).to(joint_angle.device)
        right_joint_angle = joint_angle[:,::2]
        left_joint_angle = joint_angle[:,1::2]

        tg_batch_right = self.right_robot.forward_kinematics(right_joint_angle, end_only=False)
        tg_batch_left = self.left_robot.forward_kinematics(left_joint_angle, end_only=False)
        for idx, link_name in enumerate(self.link_names):
            if 'right' in link_name:
                tg_batch = tg_batch_right
            else:
                tg_batch = tg_batch_left
            t[:,idx] = tg_batch[link_name].get_matrix()[:,:3,3]
            R[:,idx] = tg_batch[link_name].get_matrix()[:,:3,:3]
        return R, t