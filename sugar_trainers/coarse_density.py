import os
import numpy as np
import torch
import open3d as o3d
from PIL import Image
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from sugar_scene.gs_model import GaussianSplattingWrapper, fetchPly
from sugar_scene.sugar_model import SuGaR
from sugar_scene.sugar_optimizer import OptimizationParams, SuGaROptimizer
from sugar_scene.sugar_densifier import SuGaRDensifier
from sugar_utils.loss_utils import ssim, l1_loss, l2_loss

from rich.console import Console
import time


def coarse_training_with_density_regularization(args):
    CONSOLE = Console(width=120)

    # ✅ `gt_alpha_mask` 및 `gt_alpha_mask_path` 추가
    use_gt_alpha_mask = args.gt_alpha_mask  
    gt_alpha_mask_path = args.gt_alpha_mask_path  

    # ==================== Model & Training Parameters ====================

    num_device = args.gpu
    detect_anomaly = False
    num_iterations = 15_000  # Default

    # Background tensor 설정
    use_white_background = args.white_background
    bg_tensor = torch.ones(3, dtype=torch.float, device=f'cuda:{num_device}') if use_white_background else None

    # ----- Loss Function -----
    loss_function = 'l1+dssim'  # Default
    dssim_factor = 0.2 if loss_function == 'l1+dssim' else None

    # ----- Device Setup -----
    torch.cuda.set_device(num_device)
    device = torch.device(f'cuda:{num_device}')
    CONSOLE.print(f"Using device: {num_device}")
    torch.autograd.set_detect_anomaly(detect_anomaly)

    # Checkpoint 저장 경로 설정
    sugar_checkpoint_path = os.path.join(args.output_dir, f'sugarcoarse_{args.iteration_to_load}/')
    os.makedirs(sugar_checkpoint_path, exist_ok=True)

    # ==================== Load NeRF Model and Training Data ====================
    CONSOLE.print(f"\nLoading Gaussian Splatting checkpoint from: {args.checkpoint_path}...")
    nerfmodel = GaussianSplattingWrapper(
        source_path=args.scene_path,
        output_path=args.checkpoint_path,
        iteration_to_load=args.iteration_to_load,
        load_gt_images=True,
        eval_split=args.eval,
        eval_split_interval=8,  # Default eval split interval
        white_background=use_white_background,
    )

    CONSOLE.print(f"Training images detected: {len(nerfmodel.training_cameras)}")

    # ==================== Initialize SuGaR Model ====================
    CONSOLE.print("Initializing SuGaR model...")
    sugar = SuGaR(
        nerfmodel=nerfmodel,
        points=nerfmodel.gaussians.get_xyz.detach().float().cuda(),
        colors=nerfmodel.gaussians.get_features[:, 0].detach().float().cuda(),
        initialize=True,
    )

    CONSOLE.print(f"SuGaR model initialized with {len(sugar.points)} points.")

    # ==================== Initialize Optimizer ====================
    optimizer = SuGaROptimizer(
        sugar,
        OptimizationParams(iterations=num_iterations),
        spatial_lr_scale=sugar.get_cameras_spatial_extent()
    )
    
    # ==================== Loss Function Setup ====================
    def loss_fn(pred_rgb, gt_rgb):
        if loss_function == 'l1':
            return l1_loss(pred_rgb, gt_rgb)
        elif loss_function == 'l2':
            return l2_loss(pred_rgb, gt_rgb)
        elif loss_function == 'l1+dssim':
            return (1.0 - dssim_factor) * l1_loss(pred_rgb, gt_rgb) + dssim_factor * (1.0 - ssim(pred_rgb, gt_rgb))
    
    CONSOLE.print(f'Using loss function: {loss_function}')

    # ✅ `gt_alpha_mask_path`에서 알파 마스크 로드하는 함수
    def load_gt_alpha_mask(image_name):
        """gt_alpha_mask_path에서 알파 마스크 이미지를 로드하는 함수"""
        alpha_mask_file = os.path.join(gt_alpha_mask_path, f"{image_name}.png")  # PNG 확장자 가정
        if os.path.exists(alpha_mask_file):
            alpha_mask = torch.tensor(np.array(Image.open(alpha_mask_file).convert("L")) / 255.0, dtype=torch.float32)
            return alpha_mask.unsqueeze(0)  # (1, H, W) 형태로 변환
        else:
            return None

    # ==================== Start Training ====================
    sugar.train()
    train_losses = []
    t0 = time.time()
    iteration = 0

    for batch in range(9_999_999):
        if iteration >= num_iterations:
            break

        shuffled_idx = torch.randperm(len(nerfmodel.training_cameras))
        train_num_images = len(shuffled_idx)

        for i in range(0, train_num_images, 1):
            iteration += 1
            optimizer.update_learning_rate(iteration)

            camera_indices = shuffled_idx[i:i+1]

            # ✅ 예측 이미지 생성 (Gaussian Splatting)
            outputs = sugar.render_image_gaussian_rasterizer(
                camera_indices=camera_indices.item(),
                bg_color=bg_tensor,
                sh_deg=3,
                compute_color_in_rasterizer=True,
            )
            pred_rgb = outputs['image'].view(-1, sugar.image_height, sugar.image_width, 3)
            pred_rgb = pred_rgb.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)

            # ✅ GT 이미지 + 이미지 이름 가져오기
            gt_image, image_name = nerfmodel.get_gt_image(camera_indices=camera_indices, return_name=True)
            gt_rgb = gt_image.view(-1, sugar.image_height, sugar.image_width, 3)
            gt_rgb = gt_rgb.permute(0, 3, 1, 2)

            # ✅ 배경을 제외할 알파 마스크 로드
            if use_gt_alpha_mask and gt_alpha_mask_path is not None:
                gt_alpha_mask = load_gt_alpha_mask(image_name)
                if gt_alpha_mask is None:
                    gt_alpha_mask = torch.ones_like(gt_rgb[:, :1, :, :])  # ✅ 모든 픽셀을 1로 설정 (배경 제거 X)
            else:
                gt_alpha_mask = torch.ones_like(gt_rgb[:, :1, :, :])  # ✅ 기본값

            masked_pred = pred_rgb * gt_alpha_mask
            masked_gt = gt_rgb * gt_alpha_mask

            loss = loss_fn(masked_pred, masked_gt)  # ✅ 배경을 제외한 손실 계산

            # Loss Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Print Loss
            if iteration % 200 == 0:
                train_losses.append(loss.detach().item())
                CONSOLE.print(f"Iteration: {iteration}/{num_iterations}, Loss: {loss.item():.6f}, Time: {(time.time() - t0) / 60:.2f} min")
                t0 = time.time()

            if iteration >= num_iterations:
                break

    # Save Final Model
    final_model_path = os.path.join(sugar_checkpoint_path, f'final_{iteration}.pt')
    sugar.save_model(final_model_path)
    CONSOLE.print(f"Training finished. Final model saved at {final_model_path}")

    return final_model_path
