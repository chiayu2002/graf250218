import argparse
from argparse import Namespace
from submodules.nerf_pytorch.run_nerf_mod import create_nerf
import numpy as np
import os
import time
import copy
import torch.optim as optim
from tqdm import tqdm
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import wandb
import sys
sys.path.append('submodules')

from graf.gan_training import Evaluator
from graf.config import get_data, build_models, load_config, save_config
from graf.utils import get_zdist
from graf.get_poses import sample_select_pose 
from GAN_stability.gan_training.checkpoints_mod import CheckpointIO
from submodules.nerf_pytorch.run_nerf_helpers_mod import get_rays
from submodules.nerf_pytorch.run_nerf_mod import render

def setup_directories(config):
    out_dir = os.path.join(config['training']['outdir'], config['expname'])
    checkpoint_dir = os.path.join(out_dir, 'chkpts')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return out_dir, checkpoint_dir

def initialize_training(config, device):
    # dataset
    train_dataset, hwfr, K= get_data(config)
    if config['data']['orthographic']:
        hw_ortho = (config['data']['far']-config['data']['near'],) * 2
        hwfr[2] = hw_ortho
    config['data']['hwfr'] = hwfr
    config['data']['K'] = K
    
    # train_loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['nworkers'],
        shuffle=True, 
        pin_memory=True,
        sampler=None, 
        drop_last=True,
        generator=torch.Generator(device='cuda:0')
    )
    
    # Create models
    generator, discriminator, render_kwargs_train = build_models(config)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    return train_loader, generator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    
    # load config
    config = load_config(args.config)
    config['data']['fov'] = float(config['data']['fov'])
    restart_every = config['training']['restart_every']
    batch_size=config['training']['batch_size']
    fid_every = config['training']['fid_every']
    device = torch.device("cuda:0")
    
    # 創建目錄
    out_dir, checkpoint_dir = setup_directories(config)
    save_config(os.path.join(out_dir, 'config.yaml'), config)
    
    # 初始化model
    train_loader, generator = initialize_training(config, device)
    config_nerf = Namespace(**config['nerf'])
    # Update config for NERF
    config_nerf.chunk = min(config['training']['chunk'], 1024*config['training']['batch_size'])     # let batch size for training with patches limit the maximal memory
    config_nerf.netchunk = config['training']['netchunk']
    config_nerf.feat_dim = config['z_dist']['dim']
    config_nerf.num_class = config['discriminator']['num_classes']

    render_kwargs_train, render_kwargs_test, params, start  = create_nerf(config_nerf)
    global_step = start

    bds_dict = {'near': config['data']['near'], 'far': config['data']['far']}
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    
    # 優化器
    optimizer = torch.optim.Adam(params=params, lr=0.0005, betas=(0.9, 0.999))


    #get patch
    hwfr = config['data']['hwfr']
    H=W = hwfr[0]
    focal = hwfr[2]
    # img_to_patch = ImgToPatch(generator.ray_sampler, hwfr[:3])
    
    # 初始化 wandb
    wandb.init(
        project="graf250218",
        entity="vicky20020808",
        name="RS315 select",
        config=config
    )
    
    # 設置檢查點
    checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)
    checkpoint_io.register_modules(
        optimizer=optimizer,
        **generator.module_dict
    )
    
    
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)
    img2mse = lambda x, y : torch.mean((x - y) ** 2)
    mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

    # Evaluator
    evaluator = Evaluator(fid_every > 0, generator, zdist, None,
                          batch_size=batch_size, device=device, inception_nsamples=33)
    
    it = epoch_idx = -1
    t0 = time.time()
    
    while True:
        start += 1
        for x_real, label in tqdm(train_loader, desc=f"Epoch {start}"):
            it += 1
            
            label = label.to(device)
            v = config['data']['v']
            all_poses = []
            v_list = [float(x.strip()) for x in v.split(",")]
            for i in range(label.size(0)):
                second_value = label[i, 1].item()
                index = int(label[i, 2].item())  # 得到第3個值
                selected_u = index/360
                if second_value == 0:
                    pose = sample_select_pose(selected_u, v_list[0])
                elif second_value == 1:
                    pose = sample_select_pose(selected_u, v_list[1])
                elif second_value == 2:
                    pose = sample_select_pose(selected_u, v_list[2])
                elif second_value == 3:
                    pose = sample_select_pose(selected_u, v_list[3])
                else:
                    pose = sample_select_pose(selected_u, v_list[4])
                all_poses.append(pose)
            poses = torch.stack(all_poses, dim=0)

            idx = np.random.choice(8)  # 从8个样本中随机选择一个
            # img_i = x_real[idx]
            target = x_real[idx]
            target = target.to(device)
            pose = poses[idx, :3,:4]
            

            rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))

            if it < 500:
                # 在初始迭代中使用中心裁剪
                dH = int(H//2 * 0.5)
                dW = int(W//2 * 0.5)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                        torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                    ), -1)
            else:
                # 使用全图随机采样
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(0, H-1, H), 
                        torch.linspace(0, W-1, W)
                    ), -1)  # (H, W, 2)
            
            # 将坐标展平并随机选择N_rand个点
            coords = torch.reshape(coords, [-1,2]) # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[1024], replace=False)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            # select_coords[:, 0] = torch.clamp(select_coords[:, 0], 0, H-1)
            # select_coords[:, 1] = torch.clamp(select_coords[:, 1], 0, W-1)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_s = target[:, select_coords[:, 0], select_coords[:, 1]].permute(1, 0)
            print(f"rays_o shape: {rays_o.shape}, select_coords range: {select_coords.min().item()}, {select_coords.max().item()}")

            # 确保所有张量在同一设备上
            print(f"Device - rays_o: {rays_o.device}, select_coords: {select_coords.device}")
            ray_directions = batch_rays[1]  # rays_d
            norm = torch.norm(ray_directions, dim=-1)
            print(f"Min ray direction norm: {norm.min().item()}")
            if (norm < 1e-6).any():
                print("Warning: Some ray directions have very small norms!")

            # batch_size = x_real.shape[0]
            # target_s = []

            # for b in range(batch_size):
            #     # 對每個批次中的圖像提取選定座標的像素值
            #     pixels = x_real[b, :, select_coords[:, 0], select_coords[:, 1]]  # [channels, N_rand]
            #     target_s.append(pixels.transpose(0, 1))  # 轉置為 [N_rand, channels]

            # 將所有批次的結果堆疊起來
            # target_s = torch.cat(target_s, dim=0)  # [batch_size * N_rand, channels]

            rgb, _, _, _ = render(H, W, focal, label, chunk=65536, rays=batch_rays, **render_kwargs_train)

            # # Generators updates
            # if config['nerf']['decrease_noise']:
            #     generator.decrease_nerf_noise(it)

            optimizer.zero_grad()

            img_loss = img2mse(rgb, target_s)
            loss = img_loss
            # psnr = mse2psnr(img_loss)
            loss.backward()
            optimizer.step()

            decay_rate = 0.1
            decay_steps = 250 * 1000
            new_lrate =0.0005 * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
                
            # wandb
            if (it + 1) % config['training']['print_every'] == 0:
                wandb.log({
                    # "loss/discriminator": dloss,
                    "loss/generator": loss,
                    # "loss/regularizer": reg,
                    "iteration": it
                })

            # (ii) Sample if necessary
            if ((it % config['training']['sample_every']) == 0) or ((it < 500) and (it % 100 == 0)):

                plist = []
                angle_positions = [(i/8, 0.5) for i in range(8)] 
                ztest = zdist.sample((batch_size,))
                label_test = torch.tensor([[0] if i < 4 else [0] for i in range(batch_size)])
                for i, (u, v) in enumerate(angle_positions):
                    # position_angle = (azimuth + 180) % 360
                    # print(f"指定角度:{u}, 轉換後角度:{position_angle}")
                    poses = generator.sample_select_pose(u ,v)
                    plist.append(poses)
                    ptest = torch.stack(plist)

                rgb, depth, acc = evaluator.create_samples(ztest.to(device), label_test, ptest)
                    
                wandb.log({
                    "sample/rgb": [wandb.Image(rgb, caption=f"RGB at iter {it}")],
                    "sample/depth": [wandb.Image(depth, caption=f"Depth at iter {it}")],
                    "sample/acc": [wandb.Image(acc, caption=f"Acc at iter {it}")],
                    "epoch_idx": epoch_idx,
                    "iteration": it
                })

            # 儲存檢查點
            if time.time() - t0 > config['training']['save_every']:
                checkpoint_io.save(
                    config['training']['model_file'], 
                    it=it, 
                    epoch_idx=epoch_idx,
                    save_to_wandb=True
                )
                t0 = time.time()
                
                if (restart_every > 0 and t0 - tstart > restart_every):
                    return

if __name__ == '__main__':
    main()