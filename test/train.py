import argparse
import os
import time
import copy
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

# 設定所有隨機種子的函數，確保訓練的完全可重現性
def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_directories(config):
    out_dir = os.path.join(config['training']['outdir'], config['expname'])
    checkpoint_dir = os.path.join(out_dir, 'chkpts')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return out_dir, checkpoint_dir

def initialize_training(config, device, fixed_seed=42):
    # dataset
    train_dataset, hwfr= get_data(config)
    if config['data']['orthographic']:
        hw_ortho = (config['data']['far']-config['data']['near'],) * 2
        hwfr[2] = hw_ortho
    config['data']['hwfr'] = hwfr
    
    # 修改 train_loader: 關閉 shuffle，使用固定種子的 generator
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['nworkers'],
        shuffle=False,  # 關閉 shuffle 以確保數據順序一致 
        pin_memory=True,
        sampler=None, 
        drop_last=True,
        generator=torch.Generator(device='cuda:0').manual_seed(fixed_seed)
    )
    
    # Create models
    generator, discriminator = build_models(config)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # 初始化固定參考視角
    initialize_canonical_view(generator)
    
    return train_loader, generator, discriminator

def initialize_canonical_view(generator):
    """為生成器初始化標準參考視角"""
    # 設定固定的參考視角
    generator.canonical_u = 0.0  # 水平角度 (0-1)
    generator.canonical_v = 0.5  # 垂直角度 (接近赤道平面)
    
    # 產生標準視角的相機姿態
    canonical_pose = generator.sample_select_pose(generator.canonical_u, generator.canonical_v)
    generator.canonical_pose = canonical_pose
    
    # 記錄該視角下的射線
    sampler = generator.val_ray_sampler
    canonical_rays, _, _ = sampler(generator.H, generator.W, generator.focal, canonical_pose)
    generator.canonical_rays = canonical_rays
    
    # 生成參考視角的一些基準信息
    with torch.no_grad():
        # 這裡可以額外記錄一些參考信息，如果需要的話
        pass

def save_with_config(checkpoint_io, filename, it, epoch_idx, generator, save_to_wandb=False):
    """保存檢查點時包含標準視角配置"""
    # 原始保存邏輯
    checkpoint_io.save(filename, it=it, epoch_idx=epoch_idx, save_to_wandb=save_to_wandb)
    
    # 額外保存標準視角的配置
    config_path = os.path.join(os.path.dirname(checkpoint_io.checkpoint_dir), 'canonical_config.pt')
    torch.save({
        'canonical_u': generator.canonical_u,
        'canonical_v': generator.canonical_v,
        'canonical_pose': generator.canonical_pose
    }, config_path)

def main():
    # 在主函數開頭設置所有隨機種子，確保可重現性
    fixed_seed = 42
    set_all_seeds(fixed_seed)
    
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
    
    # 初始化model，傳入固定種子
    train_loader, generator, discriminator = initialize_training(config, device, fixed_seed)
    
    # 優化器
    lr_g = config['training']['lr_g']
    lr_d = config['training']['lr_d']
    g_params = generator.parameters()
    d_params = discriminator.parameters()
    
    # 使用確定性的優化器初始化
    g_optimizer = optim.RMSprop(g_params, lr=lr_g, alpha=0.99, eps=1e-8)
    d_optimizer = optim.RMSprop(d_params, lr=lr_d, alpha=0.99, eps=1e-8)

    #get patch
    hwfr = config['data']['hwfr']
    img_to_patch = ImgToPatch(generator.ray_sampler, hwfr[:3])
    
    # 初始化 wandb
    wandb.init(
        project="graf250218",
        entity="vicky20020808",
        name="RS315 fixed_angles",  # 更新名稱以反映角度固定
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
    
    # 使用固定種子初始化分布
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

            d_real = discriminator(rgbs, label)
            dloss_real = compute_loss(d_real, 1)
            reg = 10. * compute_grad2(d_real, rgbs).mean()
            
            # 使用固定種子生成隨機向量
            torch.manual_seed(fixed_seed + it)  # 確保每個迭代的隨機性一致但不同
            z = zdist.sample((batch_size,))
            
            x_fake = generator(z, label)
            d_fake = discriminator(x_fake, label)
            dloss_fake = compute_loss(d_fake, 0)

            dloss = dloss_real + dloss_fake
            dloss_all = dloss_real + dloss_fake + reg
            dloss_all.backward()
            d_optimizer.step()

            # Generators updates
            if config['nerf']['decrease_noise']:
                generator.decrease_nerf_noise(it)

            g_optimizer.zero_grad()

            # 再次使用相同的種子以確保一致性
            torch.manual_seed(fixed_seed + it)
            z = zdist.sample((batch_size,))
            
            x_fake = generator(z, label)
            d_fake = discriminator(x_fake, label)

            gloss = compute_loss(d_fake, 1)
            gloss.backward()
            g_optimizer.step()
                
            # wandb
            if (it + 1) % config['training']['print_every'] == 0:
                wandb.log({
                    "loss/discriminator": dloss,
                    "loss/generator": gloss,
                    "loss/regularizer": reg,
                    "iteration": it
                })

            # (ii) Sample if necessary
            if ((it % config['training']['sample_every']) == 0) or ((it < 500) and (it % 100 == 0)):
                # 使用固定種子生成測試樣本
                torch.manual_seed(fixed_seed)
                
                plist = []
                # 首先添加標準參考視角，確保一致性
                # ref_pose = generator.sample_select_pose(generator.canonical_u, generator.canonical_v)
                # plist.append(ref_pose)
                
                # 然後添加其他視角供視覺化
                angle_positions = [(i/8, 0.5) for i in range(8)] 
                ztest = zdist.sample((batch_size,))
                label_test = torch.tensor([[0] if i < 4 else [0] for i in range(batch_size)])
                
                for i, (u, v) in enumerate(angle_positions):
                    poses = generator.sample_select_pose(u, v)
                    plist.append(poses)
                
                ptest = torch.stack(plist)
                rgb, depth, acc = evaluator.create_samples(ztest.to(device), label_test, ptest)
                    
                wandb.log({
                    "sample/rgb": [wandb.Image(rgb, caption=f"RGB at iter {it}, first is reference")],
                    "sample/depth": [wandb.Image(depth, caption=f"Depth at iter {it}")],
                    "sample/acc": [wandb.Image(acc, caption=f"Acc at iter {it}")],
                    "epoch_idx": epoch_idx,
                    "iteration": it
                })

            # (i) Backup if necessary
            if ((it + 1) % 50000) == 0:
                print('Saving backup...')
                save_with_config(checkpoint_io, 'model_%08d.pt' % it, it, epoch_idx, generator, True)

            # 儲存檢查點
            if time.time() - t0 > config['training']['save_every']:
                save_with_config(
                    checkpoint_io,
                    config['training']['model_file'], 
                    it, 
                    epoch_idx,
                    generator,
                    True
                )
                t0 = time.time()
                
                if (restart_every > 0 and t0 - tstart > restart_every):
                    return

if __name__ == '__main__':
    main()