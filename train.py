import argparse
import os
import time
import torch.nn.functional as F
import random
import numpy as np
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
from graf.train_step import compute_grad2, compute_loss
from graf.transforms import ImgToPatch
 
from GAN_stability.gan_training.checkpoints_mod import CheckpointIO

def setup_directories(config):
    out_dir = os.path.join(config['training']['outdir'], config['expname'])
    checkpoint_dir = os.path.join(out_dir, 'chkpts')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return out_dir, checkpoint_dir

def initialize_training(config, device):
    # dataset
    train_dataset, hwfr= get_data(config)
    if config['data']['orthographic']:
        hw_ortho = (config['data']['far']-config['data']['near'],) * 2
        hwfr[2] = hw_ortho
    config['data']['hwfr'] = hwfr
    
    # train_loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['nworkers'],
        shuffle=False, 
        pin_memory=True,
        sampler=None, 
        drop_last=True,
        generator=torch.Generator(device='cuda:0')
    )
    
    # Create models
    generator, discriminator = build_models(config)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    return train_loader, generator, discriminator

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
    train_loader, generator, discriminator = initialize_training(config, device)
    
    # 優化器
    lr_g = config['training']['lr_g']
    lr_d = config['training']['lr_d']
    g_params = generator.parameters()
    d_params = discriminator.parameters()
    g_optimizer = optim.RMSprop(g_params, lr=lr_g, alpha=0.99, eps=1e-8)
    d_optimizer = optim.RMSprop(d_params, lr=lr_d, alpha=0.99, eps=1e-8)

    #get patch
    hwfr = config['data']['hwfr']
    img_to_patch = ImgToPatch(generator.ray_sampler, hwfr[:3])
    
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
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        **generator.module_dict
    )
    
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)

    # Evaluator
    evaluator = Evaluator(fid_every > 0, generator, zdist, None,
                          batch_size=batch_size, device=device, inception_nsamples=33)
    
    it = epoch_idx = -1
    tstart = t0 = time.time()
    
    while True:
        epoch_idx += 1
        for x_real, label in tqdm(train_loader, desc=f"Epoch {epoch_idx}"):
            it += 1
            
            generator.ray_sampler.iterations = it
            generator.train()
            discriminator.train()

            # Discriminator updates
            d_optimizer.zero_grad()

            x_real = x_real.to(device)
            rgbs = img_to_patch(x_real)
            rgbs.requires_grad_(True)

            z = zdist.sample((batch_size,))
            x_fake, rays, angles = generator(z, label)
            # rays_first = rays[0]
            # real_tensor = torch.cat([rgbs, rays_first], dim=1)

            d_real = discriminator(rgbs, label)
            dloss_real = compute_loss(d_real, 1)
            reg = 10. * compute_grad2(d_real, rgbs).mean()
            
            # angles_first = angles[:, :3]
            # fake_tensor = torch.cat([x_fake, angles_first], dim=1)
            d_fake = discriminator(x_fake, label)
            dloss_fake = compute_loss(d_fake, 0)

            dloss = dloss_real + dloss_fake
            dloss_all = dloss_real + dloss_fake +reg
            dloss_all.backward()
            d_optimizer.step()

            # Generators updates
            if config['nerf']['decrease_noise']:
                generator.decrease_nerf_noise(it)

            g_optimizer.zero_grad()

            z = zdist.sample((batch_size,))
            x_fake, rays, angles = generator(z, label)
            reshaped_rays = torch.cat([rays[0], rays[1]], dim=1)
            # rays_second_dim = rays[1] 
            # angles_second = angles[:, 3:]
            mse_loss = F.mse_loss(reshaped_rays, angles)
            # angles_first = angles[:, :3]
            # fake_tensor = torch.cat([x_fake, angles_first], dim=1)
            d_fake = discriminator(x_fake, label)

            gloss = compute_loss(d_fake, 1) 
            all_loss = gloss + mse_loss

            all_loss.backward()
            g_optimizer.step()
                
            # wandb
            if (it + 1) % config['training']['print_every'] == 0:
                wandb.log({
                    "loss/discriminator": dloss,
                    "loss/generator": gloss,
                    "loss/mse": mse_loss,
                    "loss/regularizer": reg,
                    "iteration": it
                })

            # (ii) Sample if necessary
            if ((it % config['training']['sample_every']) == 0) or ((it < 500) and (it % 100 == 0)):
                # torch.manual_seed(42)
                plist = []
                angle_positions = [(i/8, 0.5) for i in range(8)] 
                ztest = zdist.sample((batch_size,))
                label_test = torch.tensor([[0] if i < 4 else [0] for i in range(batch_size)])
                for i, (u, v) in enumerate(angle_positions):
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

            # (i) Backup if necessary
            if ((it + 1) % 50000) == 0:
                print('Saving backup...')
                checkpoint_io.save('model_%08d.pt' % it, it=it, epoch_idx=epoch_idx, save_to_wandb=True)

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