import numpy as np
import torch
import pprint
import yaml
from tqdm import tqdm
import wandb

from datasets.dream import DREAM
from datasets.transforms import make_transforms
from datasets.collator import MaskCollator
from models.model import make_robotposenet
from models.losses import FocalHeatmapLoss
from datasets.image_proc import get_keypoints_from_beliefmap
from accelerate import Accelerator, DistributedDataParallelKwargs


def main(args):
    device = 'cuda'
    robot_name = args['robot']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    batch_size = args['data']['batch_size']
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
    epochs = args['training']['epochs']
    pred_emb_dim = args['model']['pred_emb_dim']
    pred_depth = args['model']['pred_depth']
    urdf_file = args['model']['urdf_file']
    log_wandb = args['training']['log_wandb']
    max_lr = [float(x) for x in args['training']['max_lr']]
    weight_decay = float(args['training']['weight_decay'])
    final_div_factor = float(args['training']['final_div_factor'])
    use_accelerate = args['training']['use_accelerate']

    if use_accelerate:
        accelerator = Accelerator()

    if log_wandb and use_accelerate and accelerator.is_local_main_process:
        wandb.login()
        name = "Trial_"+str(np.random.randint(1000))
        run = wandb.init(
            # Set the project where this run will be logged
            project="RoboPEPP",
            # Track hyperparameters and run metadata
            name=name,
        )


    mask_collator = MaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=(0.0,0.2),
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

    train_dataset = DREAM(
        root=root_path,
        image_folder=image_folder,
        transform=transform,
        train=True,
        index_targets=False,
        crop_size = crop_size,
        robot_name=robot_name)
    
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

    val_dataset = DREAM(
        root=root_path,
        image_folder=image_folder,
        transform=val_transform,
        train=False,
        index_targets=False,
        crop_size = crop_size,
        robot_name=robot_name)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=mask_collator,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=False)
    
    if robot_name=="panda":
        joint_mean = torch.Tensor([-5.21536266e-02, 2.67705064e-01,  6.04037923e-03, -2.00518761e+00,
                                1.49018339e-02,  1.98561130e+00]).to(device)
        joint_std = torch.Tensor([1.02490399e+00, 6.45399420e-01, 5.11071070e-01, 5.08277903e-01,
                                7.68895279e-01, 5.10862160e-01]).to(device)
        model = make_robotposenet(backbone, input_shape=(image_shape, image_shape), patch_size=patch_size, 
                                r_path=r_path, pred_emb_dim=pred_emb_dim, pred_depth=pred_depth, 
                                joint_mean=joint_mean, joint_std=joint_std, npose=6, nkeypoints=7, urdf_file=urdf_file).to(device)
    elif robot_name=="kuka":
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
    
    optimizer = torch.optim.AdamW([
        {'params': model.context_backbone.parameters(), 'lr': 1e-4},
        {'params': model.predictor_backbone.parameters(), 'lr': 1e-4},
        {'params': model.joint_net.parameters(), 'lr': 1e-4},
        {'params': model.keypoint_net.parameters(), 'lr': 1e-4},
    ], weight_decay=weight_decay)

    # Define the learning rate scheduler with the same optimizer
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        pct_start=0.0,
        final_div_factor=final_div_factor,
        max_lr=max_lr,  # Different max_lr for each parameter group
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

    for epoch in range(epochs):
        train_loader.dataset.set_epoch(epoch)
        val_loader.dataset.set_epoch(epoch)
        model.train()
        if robot_name=="panda":
            train_l1loss_joints = np.zeros([1,6])
            train_l1loss_keypoints = np.zeros([1,7])
            train_l1loss_3d_keypoints = np.zeros([1,7])
        elif robot_name=="kuka":
            train_l1loss_joints = np.zeros([1,6])
            train_l1loss_keypoints = np.zeros([1,8])
            train_l1loss_3d_keypoints = np.zeros([1,8])
        elif robot_name=="baxter":
            train_l1loss_joints = np.zeros([1,12])
            train_l1loss_keypoints = np.zeros([1,17])
            train_l1loss_3d_keypoints = np.zeros([1,17])
        train_loss_joints = 0
        train_loss_keypoints = 0
        if use_accelerate:
            if accelerator.is_local_main_process:
                print(f"Epoch {epoch}")
                iterator = tqdm(train_loader, dynamic_ncols=True)
            else:
                iterator = train_loader
        else:
            print(f"Epoch {epoch}")
            iterator = tqdm(train_loader, dynamic_ncols=True)
        if epoch<5:
            alpha1 = 1e-4
        elif epoch<10:
            alpha1 = 1e-2
        elif epoch<40:
            alpha1 = 1e-1
        else:
            alpha1 = 1
        dist3d_train = []
        for i, ((x, gt_joints, gt_keypoints, metadata), masks_enc, masks_pred) in enumerate(iterator):
            x, gt_joints, gt_keypoints = x.to(device, non_blocking=True), gt_joints.to(device, non_blocking=True), gt_keypoints.to(device, non_blocking=True)
            if robot_name=="panda" or robot_name=="kuka":
                gt_joints = gt_joints[:,:6]
            elif robot_name=="baxter":
                gt_joints = gt_joints[:,1:13]
            gt_joints = (gt_joints - joint_mean) / joint_std

            masks_enc = [u.to(device, non_blocking=True) for u in masks_enc]
            masks_pred = [u.to(device, non_blocking=True) for u in masks_pred]

            gt_kp_3d_cam = metadata['orig_keypoints_3d'].to(device, non_blocking=True)
            K = metadata['K'].to(device, non_blocking=True)
            valid_indices_mask = metadata['valid_indices_mask'].to(device, non_blocking=True)
            valid_indices_mask_uv = valid_indices_mask.unsqueeze(-1).repeat(1,1,2)
            valid_indices_mask = valid_indices_mask.unsqueeze(-1).unsqueeze(-1).repeat(1,1,224,224)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred_joints, pred_keypoints, pred_kp_3d_cam, _, _ = model(x, K, metadata, masks_enc, masks_pred)
                l1loss_joints = criterion_l1(pred_joints*joint_std + joint_mean, gt_joints*joint_std + joint_mean)
                l1loss_keypoints = criterion_l1(get_keypoints_from_beliefmap(pred_keypoints), get_keypoints_from_beliefmap(gt_keypoints))*valid_indices_mask_uv
                l1loss_3d_keypoints = criterion_l1(pred_kp_3d_cam, gt_kp_3d_cam)

                loss_joints = criterion_mse(pred_joints, gt_joints).mean()

                dist = torch.norm(pred_kp_3d_cam - gt_kp_3d_cam, dim=-1)
                dist = dist.view(-1).detach().cpu().numpy().tolist()
                dist3d_train.extend(dist)
                loss_keypoints = (criterion_focal(pred_keypoints, gt_keypoints)*valid_indices_mask).mean()

                loss = loss_joints + alpha1*loss_keypoints

                train_loss_joints += loss_joints.detach().cpu().numpy()
                train_loss_keypoints += loss_keypoints.detach().cpu().numpy()
            if use_accelerate:
                accelerator.backward(scaler.scale(loss))
            else:
                scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            
            
            train_l1loss_joints += l1loss_joints.mean(dim=0).detach().cpu().numpy()
            train_l1loss_keypoints += l1loss_keypoints.mean(dim=(0,2)).detach().cpu().numpy()
            train_l1loss_3d_keypoints += l1loss_3d_keypoints.mean(dim=(0,2)).detach().cpu().numpy()

        model.eval()
        if robot_name=="panda":
            val_l1loss_joints = np.zeros([1,6])
            val_l1loss_keypoints = np.zeros([1,7])
            val_l1loss_3d_keypoints = np.zeros([1,7])
        elif robot_name=="kuka":
            val_l1loss_joints = np.zeros([1,6])
            val_l1loss_keypoints = np.zeros([1,8])
            val_l1loss_3d_keypoints = np.zeros([1,8])
        elif robot_name=="baxter":
            val_l1loss_joints = np.zeros([1,12])
            val_l1loss_keypoints = np.zeros([1,17])
            val_l1loss_3d_keypoints = np.zeros([1,17])
        
        val_loss_joints = 0
        val_loss_keypoints = 0
        if use_accelerate:
            if accelerator.is_local_main_process:
                iterator = tqdm(val_loader, dynamic_ncols=True)
            else:
                iterator = val_loader
        else:
            iterator = tqdm(val_loader, dynamic_ncols=True)

        dist3d_val = []
        with torch.inference_mode():
            for i, ((x, gt_joints, gt_keypoints, metadata), masks_enc, masks_pred) in enumerate(iterator):
                x, gt_joints, gt_keypoints = x.to(device, non_blocking=True), gt_joints.to(device, non_blocking=True), gt_keypoints.to(device, non_blocking=True)
                if robot_name=="panda" or robot_name=="kuka":
                    gt_joints = gt_joints[:,:6]
                elif robot=="baxter":
                    gt_joints = gt_joints[:,1:13]
                masks_enc = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_pred = [u.to(device, non_blocking=True) for u in masks_pred]

                valid_indices_mask = metadata['valid_indices_mask'].to(device, non_blocking=True)
                valid_indices_mask_uv = valid_indices_mask.unsqueeze(-1).repeat(1,1,2)
                valid_indices_mask = valid_indices_mask.unsqueeze(-1).unsqueeze(-1).repeat(1,1,224,224)

                gt_kp_3d_cam = metadata['orig_keypoints_3d'].to(device, non_blocking=True)
                K = metadata['K'].to(device, non_blocking=True)

                pred_joints, pred_keypoints, pred_kp_3d_cam, _, _ = model(x, K, metadata, masks_enc, masks_pred)

                l1loss_joints = criterion_l1(pred_joints*joint_std + joint_mean, gt_joints)
                l1loss_keypoints = criterion_l1(get_keypoints_from_beliefmap(pred_keypoints), get_keypoints_from_beliefmap(gt_keypoints))*valid_indices_mask_uv
                l1loss_3d_keypoints = criterion_l1(pred_kp_3d_cam, gt_kp_3d_cam)

                dist = torch.norm(pred_kp_3d_cam - gt_kp_3d_cam, dim=-1)
                dist = dist.view(-1).detach().cpu().numpy().tolist()
                dist3d_val.extend(dist)

                val_l1loss_keypoints += l1loss_keypoints.mean(dim=(0,2)).detach().cpu().numpy()
                val_l1loss_joints += l1loss_joints.mean(dim=0).detach().cpu().numpy()
                val_l1loss_3d_keypoints += l1loss_3d_keypoints.mean(dim=(0,2)).detach().cpu().numpy()

                loss_joints = criterion_mse(pred_joints*joint_std + joint_mean, gt_joints).mean()
                loss_keypoints = (criterion_focal(pred_keypoints, gt_keypoints)*valid_indices_mask).mean()
                loss = loss_joints + 1*loss_keypoints
                val_loss_joints += loss_joints.detach().cpu().numpy()
                val_loss_keypoints += loss_keypoints.detach().cpu().numpy()
                
        


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


            val_l1loss_joints = accelerator.gather(torch.tensor(val_l1loss_joints, device=device))
            val_l1loss_joints = val_l1loss_joints.mean(dim=0).cpu().numpy()*(180/np.pi) / len(val_loader)  # Averaging over all GPUs

            val_l1loss_keypoints = accelerator.gather(torch.tensor(val_l1loss_keypoints, device=device))
            val_l1loss_keypoints = val_l1loss_keypoints.mean(dim=0).cpu().numpy() / len(val_loader)  # Averaging over all GPUs

            val_l1loss_3d_keypoints = accelerator.gather(torch.tensor(val_l1loss_3d_keypoints, device=device))
            val_l1loss_3d_keypoints = val_l1loss_3d_keypoints.mean(dim=0).cpu().numpy() / len(val_loader)  # Averaging over all GPUs

            dist3d_val = accelerator.gather(torch.tensor(dist3d_val, device=device))
            dist3d_val = dist3d_val.view(-1).detach().cpu().numpy()

            val_loss_joints = accelerator.gather(torch.tensor(val_loss_joints, device=device))
            val_loss_joints = val_loss_joints.mean(dim=0).cpu().numpy() / len(val_loader)

            val_loss_keypoints = accelerator.gather(torch.tensor(val_loss_keypoints, device=device))
            val_loss_keypoints = val_loss_keypoints.mean(dim=0).cpu().numpy() / len(val_loader)

            if accelerator.is_local_main_process:
                train_l1loss_joints_str = ', '.join([f"{loss:.2f}" for loss in train_l1loss_joints])
                train_l1loss_keypoints_str = ', '.join([f"{loss:.2f}" for loss in train_l1loss_keypoints])
                train_l1loss_3d_keypoints_str = ', '.join([f"{loss:.3f}" for loss in train_l1loss_3d_keypoints])

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

                counts_3d = []
                for value in add_threshold_values:
                    under_threshold = (
                        np.mean(dist3d_val <= value)
                    )
                    counts_3d.append(under_threshold)
                auc_add_val = np.trapz(counts_3d, dx=delta_threshold) / auc_threshold

                train_loss_joints_str = f"{train_loss_joints:.2f}"
                train_loss_keypoints_str = f"{train_loss_keypoints:.5f}"

                val_l1loss_joints_str = ', '.join([f"{loss:.2f}" for loss in val_l1loss_joints])
                val_l1loss_keypoints_str = ', '.join([f"{loss:.2f}" for loss in val_l1loss_keypoints])
                val_l1loss_3d_keypoints_str = ', '.join([f"{loss:.3f}" for loss in val_l1loss_3d_keypoints])

                val_loss_joints_str = f"{val_loss_joints:.2f}"
                val_loss_keypoints_str = f"{val_loss_keypoints:.5f}"

                current_lr = optimizer.param_groups[0]['lr']


                print(f"Training L1 Loss Joints: [{train_l1loss_joints_str}]")
                print(f"Validation L1 Loss Joints: [{val_l1loss_joints_str}]")
                print(f"Training L1 Loss Keypoints: [{train_l1loss_keypoints_str}]")
                print(f"Validation L1 Loss Keypoints: [{val_l1loss_keypoints_str}]")
                print(f"Training Loss Joints: [{train_loss_joints_str}]")
                print(f"Validation Loss Joints: [{val_loss_joints_str}]")
                print(f"Training Loss Keypoints: [{train_loss_keypoints_str}]")
                print(f"Validation Loss Keypoints: [{val_loss_keypoints_str}]")
                print(f"Training AUC 3D Keypoints: {auc_add_train:.2f}")
                print(f"Validation AUC 3D Keypoints: {auc_add_val:.2f}")

                if log_wandb:
                    wandb.log({"Current LR":current_lr, 
                               "Training loss joints": train_loss_joints, 
                               "Validation loss joints": val_loss_joints, 
                               "Training loss keypoints": train_loss_keypoints, 
                               "Validation loss keypoints": val_loss_keypoints,
                               "Training AUC 3D keypoints": auc_add_train,
                               "Validation AUC 3D keypoints": auc_add_val})
                
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict()
                }

                torch.save(checkpoint, "checkpoints/robopepp_"+str(robot_name)+".pt")
                print(f"Model, optimizer, and scheduler saved")
        else:
            train_l1loss_joints /= len(train_loader)  # Standard averaging for non-accelerate case
            train_l1loss_keypoints /= len(train_loader)  # Standard averaging for non-accelerate case
            train_l1loss_3d_keypoints /= len(train_loader)  # Standard averaging for non-accelerate case
            val_l1loss_joints /= len(val_loader)  # Standard averaging for non-accelerate case
            val_l1loss_keypoints /= len(val_loader)  # Standard averaging for non-accelerate case
            val_l1loss_3d_keypoints /= len(val_loader)  # Standard averaging for non-accelerate case
            # Format each element in the tensor to two decimal places
            train_l1loss_joints_str = ', '.join([f"{loss:.2f}" for loss in train_l1loss_joints])
            train_l1loss_keypoints_str = ', '.join([f"{loss:.2f}" for loss in train_l1loss_keypoints])
            train_l1loss_3d_keypoints_str = ', '.join([f"{loss:.2f}" for loss in train_l1loss_3d_keypoints])
            val_l1loss_joints_str = ', '.join([f"{loss:.2f}" for loss in val_l1loss_joints])
            val_l1loss_keypoints_str = ', '.join([f"{loss:.2f}" for loss in val_l1loss_keypoints])
            val_l1loss_3d_keypoints_str = ', '.join([f"{loss:.2f}" for loss in val_l1loss_3d_keypoints])

            print(f"Training Loss Joints: [{train_l1loss_joints_str}]")
            print(f"Validation Loss Joints: [{val_l1loss_joints_str}]")
            print(f"Training Loss Keypoints: [{train_l1loss_keypoints_str}]")
            print(f"Validation Loss Keypoints: [{val_l1loss_keypoints_str}]")
            print(f"Training Loss 3D Keypoints: [{train_l1loss_3d_keypoints_str}]")
            print(f"Validation Loss 3D Keypoints: [{val_l1loss_3d_keypoints_str}]")

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict()
            }

            torch.save(checkpoint, "checkpoints/robopepp_"+str(robot_name)+".pt")
            print(f"Model, optimizer, and scheduler saved")
            


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