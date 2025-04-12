import os

import numpy as np
from datasets.BPnP import BPnP_m3d
import torch # type: ignore
import argparse
import pprint
import yaml
from tqdm import tqdm
from PIL import Image
import wandb
import gc

from datasets.dream import DREAM
from datasets.dream_ssl import DREAM as DREAM_SSL
from datasets.transforms import make_transforms
from datasets.collator import MaskCollator
from models.model import make_robotposenet
from models.mesh_renderer import setup_robot_renderer, render_single_robot_mask
from models.losses import FocalHeatmapLoss
from datasets.image_proc import get_keypoints_from_beliefmap, project_to_image_plane
from accelerate import Accelerator, DistributedDataParallelKwargs # type: ignore


def scale_gradients_hook(grad):
    return grad * 1e-10

def main(args):
    device = 'cuda'
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    batch_size = args['data']['s2r_batch_size']
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
    epochs = args['training']['s2r_epochs']
    pred_emb_dim = args['model']['pred_emb_dim']
    pred_depth = args['model']['pred_depth']
    log_wandb = args['training']['log_wandb']
    s2r_seq = args['data']['s2r_seq']
    urdf_file = args['model']['urdf_file']

    use_accelerate = args['training']['use_accelerate']

    if use_accelerate:
        accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
        # accelerator = Accelerator()

    if log_wandb and use_accelerate and accelerator.is_local_main_process:
        wandb.login()
        name = "S2R_"+str(np.random.randint(1000))
        run = wandb.init(
            # Set the project where this run will be logged
            project="RoboPEPPS2R",
            # Track hyperparameters and run metadata
            name=name,
        )


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

    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter)

    train_dataset = DREAM_SSL(
        root=root_path,
        image_folder=image_folder,
        transform=transform,
        train=False,
        index_targets=False,
        crop_size = crop_size,
        test_seq=s2r_seq,
        pre_computed_bbox=True)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=mask_collator,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=False)
    
    val_transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=False,
        horizontal_flip=False,
        color_distortion=False,
        color_jitter=False)

    val_dataset = DREAM_SSL(
        root=root_path,
        image_folder=image_folder,
        transform=val_transform,
        train=False,
        index_targets=False,
        crop_size = crop_size,
        test_seq=s2r_seq,
        pre_computed_bbox=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(batch_size/1),
        collate_fn=mask_collator,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=False)
    
    joint_mean = torch.Tensor([-5.21536266e-02, 2.67705064e-01,  6.04037923e-03, -2.00518761e+00,
                            1.49018339e-02,  1.98561130e+00]).to(device)
    joint_std = torch.Tensor([1.02490399e+00, 6.45399420e-01, 5.11071070e-01, 5.08277903e-01,
                            7.68895279e-01, 5.10862160e-01]).to(device)
    
    model = make_robotposenet(backbone, input_shape=(image_shape, image_shape), patch_size=patch_size, 
                          r_path=r_path, pred_emb_dim=pred_emb_dim, pred_depth=pred_depth, 
                          joint_mean=joint_mean, joint_std=joint_std, urdf_file=urdf_file).to(device)

    checkpoint = torch.load('checkpoints/robopepp_panda.pt', map_location=torch.device('cpu'), weights_only=True)['model_state_dict']
    pretrained_dict = checkpoint
    new_state_dict = {}
    for k, v in pretrained_dict.items():
        new_key = k.replace("module.", "")  # Remove the "module." prefix
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    print("Loaded pretrained weights for the model.")
    
    optimizer = torch.optim.AdamW([
        {'params': model.context_backbone.parameters(), 'lr': 1e-7},
        {'params': model.predictor_backbone.parameters(), 'lr': 1e-7},
        {'params': model.joint_net.parameters(), 'lr': 1e-5},
        {'params': model.keypoint_net.parameters(), 'lr': 1e-5},
    ], weight_decay=1e-7)

    # Define the learning rate scheduler with the same optimizer
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        pct_start=0.0,
        final_div_factor=1e4,
        max_lr=[1e-7, 1e-7, 1e-5, 1e-10],  # Different max_lr for each parameter group
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        cycle_momentum=False
    )
    criterion_mse = torch.nn.MSELoss(reduction='none')
    criterion_l1 = torch.nn.L1Loss(reduction='none')
    criterion_focal = FocalHeatmapLoss()
    scaler = torch.amp.GradScaler('cuda')

    if use_accelerate:
        model, optimizer, train_loader, lr_scheduler, val_loader = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler, val_loader)
    
    best_auc = 0
    for epoch in range(epochs):
        model.train()
        train_l1loss_joints = np.zeros([1,6])
        train_l1loss_keypoints = np.zeros([1,7])
        train_l1loss_3d_keypoints = np.zeros([1,7])
        train_loss_joints = 0
        train_loss_keypoints = 0
        train_loss_kp_2d = 0
        if use_accelerate:
            if accelerator.is_local_main_process:
                print(f"Epoch {epoch}")
                iterator = tqdm(train_loader, dynamic_ncols=True)
            else:
                iterator = train_loader
        else:
            print(f"Epoch {epoch}")
            iterator = tqdm(train_loader, dynamic_ncols=True)

        dist3d_train = []
        for i, ((x, gt_joints, gt_keypoints, metadata), masks_enc, masks_pred) in enumerate(iterator):
            model.zero_grad()
            
            x, gt_joints, gt_keypoints = x.to(device, non_blocking=True), gt_joints.to(device, non_blocking=True), gt_keypoints.to(device, non_blocking=True)
            gt_joints = gt_joints[:,:6]
            gt_joints = (gt_joints - joint_mean) / joint_std

            masks_enc = [u.to(device, non_blocking=True) for u in masks_enc]
            masks_pred = [u.to(device, non_blocking=True) for u in masks_pred]

            masks_enc = None
            masks_pred = None

            K = metadata['K'].to(device, non_blocking=True)
            gt_kp_3d_cam = metadata['orig_keypoints_3d'].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred_joints, pred_keypoints, pred_kp_3d_cam, pred_keypoints_uv, kp_xyz_rob = model(x, K, metadata, masks_enc, masks_pred, soft_keypoints=True)

                l1loss_joints = criterion_l1(pred_joints*joint_std + joint_mean, gt_joints*joint_std + joint_mean)
                l1loss_keypoints = criterion_l1(get_keypoints_from_beliefmap(pred_keypoints), get_keypoints_from_beliefmap(gt_keypoints))
                l1loss_3d_keypoints = criterion_l1(pred_kp_3d_cam, gt_kp_3d_cam)

                loss_joints = criterion_mse(pred_joints, gt_joints).mean()

                dist = torch.norm(pred_kp_3d_cam - gt_kp_3d_cam, dim=-1)
                dist = dist.view(-1).detach().cpu().numpy().tolist()
                dist3d_train.extend(dist)
                loss_keypoints = (criterion_focal(pred_keypoints, gt_keypoints)).mean()
                # loss_keypoints = (criterion_mse(pred_keypoints, gt_keypoints)*valid_indices_mask).mean()
        
                pred_kp_2d = project_to_image_plane(pred_kp_3d_cam, K[0].float())
                # print(pred_keypoints_uv)
                loss_kp_2d = criterion_mse(pred_kp_2d, pred_keypoints_uv).mean(dim=(1,2))
                loss_kp_2d = loss_kp_2d[torch.where(loss_kp_2d < 10)].mean()

                # loss_kp_2d = criterion_mse(pred_kp_2d, pred_keypoints_uv).mean()

                pred_keypoints_uv.register_hook(scale_gradients_hook)
                # pred_kp_2d.register_hook(scale_gradients_hook)

                loss = 1 * loss_kp_2d
                

                train_loss_joints += loss_joints.detach().cpu().numpy()
                train_loss_keypoints += loss_keypoints.detach().cpu().numpy()
                train_loss_kp_2d += loss_kp_2d.detach().cpu().numpy()
            if use_accelerate:
                accelerator.backward(scaler.scale(loss))
            else:
                scaler.scale(loss).backward()
            # for param in model.module.keypoint_net.parameters():
            #     print(param.grad)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            
            
            train_l1loss_joints += l1loss_joints.mean(dim=0).detach().cpu().numpy()
            train_l1loss_keypoints += l1loss_keypoints.mean(dim=(0,2)).detach().cpu().numpy()
            train_l1loss_3d_keypoints += l1loss_3d_keypoints.mean(dim=(0,2)).detach().cpu().numpy()
            # print(train_l1loss_3d_keypoints)

            del gt_joints, gt_keypoints, x, pred_joints, pred_keypoints, pred_kp_3d_cam, \
                metadata, pred_keypoints_uv, loss
            gc.collect()
            torch.cuda.empty_cache()


        ##########################################################
        ##################### LOGGING ############################
        ##########################################################
        if use_accelerate:
            train_l1loss_joints = accelerator.gather(torch.tensor(train_l1loss_joints, device=device))
            train_l1loss_joints = train_l1loss_joints.mean(dim=0).cpu().numpy()*(180/np.pi) / len(train_loader)  # Averaging over all GPUs

            train_l1loss_keypoints = accelerator.gather(torch.tensor(train_l1loss_keypoints, device=device))
            train_l1loss_keypoints = train_l1loss_keypoints.mean(dim=0).cpu().numpy() / len(train_loader)  # Averaging over all GPUs

            train_l1loss_3d_keypoints = accelerator.gather(torch.tensor(train_l1loss_3d_keypoints, device=device))
            train_l1loss_3d_keypoints = train_l1loss_3d_keypoints.mean(dim=0).cpu().numpy() / len(train_loader)  # Averaging over all GPUs

            dist3d_train = accelerator.gather(torch.tensor(dist3d_train, device=device))
            dist3d_train = dist3d_train.view(-1).detach().cpu().numpy()

            train_loss_joints = accelerator.gather(torch.tensor(train_loss_joints, device=device))
            train_loss_joints = train_loss_joints.mean(dim=0).cpu().numpy() / len(train_loader)

            train_loss_keypoints = accelerator.gather(torch.tensor(train_loss_keypoints, device=device))
            train_loss_keypoints = train_loss_keypoints.mean(dim=0).cpu().numpy() / len(train_loader)

            train_loss_kp_2d = accelerator.gather(torch.tensor(train_loss_kp_2d, device=device))
            train_loss_kp_2d = train_loss_kp_2d.mean(dim=0).cpu().numpy() / len(train_loader)

            if accelerator.is_local_main_process:
                # print(f"Epoch {epoch}")
                # Format each element in the tensor to two decimal places
                train_l1loss_joints_str = ', '.join([f"{loss:.2f}" for loss in train_l1loss_joints])
                train_l1loss_keypoints_str = ', '.join([f"{loss:.2f}" for loss in train_l1loss_keypoints])
                train_l1loss_3d_keypoints_str = ', '.join([f"{loss:.3f}" for loss in train_l1loss_3d_keypoints])

                train_loss_joints_str = f"{train_loss_joints:.5f}"
                train_loss_keypoints_str = f"{train_loss_keypoints:.5f}"
                train_loss_kp_2d_str = f"{train_loss_kp_2d:.5f}"

                auc_threshold = 0.1
                delta_threshold = 0.00001
                add_threshold_values = np.arange(0.0, auc_threshold, delta_threshold)
                counts_3d = []
                for value in add_threshold_values:
                    under_threshold = (
                        np.mean(dist3d_train <= value)
                    )
                    counts_3d.append(under_threshold)
                auc_add_train = np.trapz(counts_3d, dx=delta_threshold) / auc_threshold

                current_lr = optimizer.param_groups[0]['lr']

                if log_wandb:
                    wandb.log({"Current LR":current_lr, 
                               "Training loss joints": train_loss_joints, 
                               "Training loss keypoints": train_loss_keypoints,
                               "Training AUC 3D keypoints": auc_add_train})
                
                # find highest AUC and save best model and recent model
                if auc_add_train > best_auc:
                    best_auc = auc_add_train
                    torch.save(model.state_dict(), "checkpoints/robopepp_s2r_"+str(s2r_seq.split('-')[1])+"_best.pt")
                    print(f"Model saved as model_best.pt")
                torch.save(model.state_dict(), "checkpoints/robopepp_s2r_"+str(s2r_seq.split('-')[1])+".pt")
                print(f"Model saved as model.pt")

                print(f"Training L1 Loss Joints: [{train_l1loss_joints_str}]")
                print(f"Training L1 Loss Keypoints: [{train_l1loss_keypoints_str}]")
                print(f"Training Loss Joints: [{train_loss_joints_str}]")
                print(f"Training Loss Keypoints: [{train_loss_keypoints_str}]")
                print(f"Training L1 Loss 3D Keypoints: [{train_l1loss_3d_keypoints_str}]")
                print(f"Training Loss Projected KP: [{train_loss_kp_2d_str}]")
                print(f"Training AUC 3D Keypoints: {auc_add_train:.5f}")
                print(f"Training Best AUC 3D Keypoints: {best_auc:.5f}")
        else:
            train_l1loss_joints /= len(train_loader)  # Standard averaging for non-accelerate case
            train_l1loss_keypoints /= len(train_loader)  # Standard averaging for non-accelerate case
            train_l1loss_3d_keypoints /= len(train_loader)  # Standard averaging for non-accelerate case
            # Format each element in the tensor to two decimal places
            train_l1loss_joints_str = ', '.join([f"{loss:.2f}" for loss in train_l1loss_joints])
            train_l1loss_keypoints_str = ', '.join([f"{loss:.2f}" for loss in train_l1loss_keypoints])
            train_l1loss_3d_keypoints_str = ', '.join([f"{loss:.2f}" for loss in train_l1loss_3d_keypoints])

            print(f"Training Loss Joints: [{train_l1loss_joints_str}]")
            print(f"Training Loss Keypoints: [{train_l1loss_keypoints_str}]")
            print(f"Training Loss 3D Keypoints: [{train_l1loss_3d_keypoints_str}]")

            torch.save(model.state_dict(), "checkpoints/robopepp_s2r_"+str(s2r_seq.split('-')[1])+".pt")
            print(f"Model saved as model.pt")
            


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