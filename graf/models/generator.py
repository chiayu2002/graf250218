import numpy as np
import torch
from ..utils import sample_on_sphere, look_at, to_sphere
from ..transforms import FullRaySampler
from submodules.nerf_pytorch.run_nerf_mod import render, run_network            # import conditional render
from functools import partial


class Generator(object):
    def __init__(self, H, W, focal, radius, ray_sampler, render_kwargs_train, render_kwargs_test, parameters, named_parameters,
                 range_u=(0,1), range_v=(0.01,0.49),v=0, chunk=None, device='cuda', orthographic=False, use_default_rays=False):
        self.device = device
        self.H = int(H)
        self.W = int(W)
        self.focal = focal
        self.radius = radius
        self.range_u = range_u
        self.range_v = range_v
        self.chunk = chunk
        self.v = v
        self.use_default_rays = use_default_rays
        coords = torch.from_numpy(np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), -1))
        self.coords = coords.view(-1, 2)

        self.ray_sampler = ray_sampler   #FlexGridRaySampler
        self.val_ray_sampler = FullRaySampler(orthographic=orthographic)
        self.render_kwargs_train = render_kwargs_train
        self.render_kwargs_test = render_kwargs_test
        self.initial_raw_noise_std = self.render_kwargs_train['raw_noise_std']
        self._parameters = parameters
        self._named_parameters = named_parameters
        self.module_dict = {'generator': self.render_kwargs_train['network_fn']}
        
        for k, v in self.module_dict.items():
            if k in ['generator']:
                continue       # parameters already included
            self._parameters += list(v.parameters())
            self._named_parameters += list(v.named_parameters())
        
        self.parameters = lambda: self._parameters           # save as function to enable calling model.parameters()
        self.named_parameters = lambda: self._named_parameters           # save as function to enable calling model.named_parameters()
        self.use_test_kwargs = False

        self.render = partial(render, H=self.H, W=self.W, focal=self.focal, chunk=self.chunk)

    def __call__(self, z, label, rays=None):
        bs = z.shape[0]
        if rays is None:
            if self.use_default_rays :
                rays = torch.cat([self.sample_rays() for _ in range(bs)], dim=1)
            else:
                all_rays = []
                v_list = [float(x.strip()) for x in self.v.split(",")]
                for i in range(label.size(0)):
                    second_value = label[i, 1].item()
                    index = int(label[i, 2].item())  # 得到第3個值
                    selected_u = index/360
                    if second_value == 0:
                        rays = torch.cat([self.sample_select_rays(selected_u, v_list[0])], dim=1)
                    elif second_value == 1:
                        rays = torch.cat([self.sample_select_rays(selected_u, v_list[1])], dim=1)
                    elif second_value == 2:
                        rays = torch.cat([self.sample_select_rays(selected_u, v_list[2])], dim=1)
                    elif second_value == 3:
                        rays = torch.cat([self.sample_select_rays(selected_u, v_list[3])], dim=1)
                    else:
                        rays = torch.cat([self.sample_select_rays(selected_u, v_list[4])], dim=1)
                    all_rays.append(rays)
                rays = torch.cat(all_rays, dim=1)


        render_kwargs = self.render_kwargs_test if self.use_test_kwargs else self.render_kwargs_train
        render_kwargs = dict(render_kwargs)        # copy

        render_kwargs['features'] = z
        rgb, disp, acc, extras = render(self.H, self.W, self.focal, label, chunk=self.chunk, rays=rays,
                                        **render_kwargs)

        rays_to_output = lambda x: x.view(len(x), -1) * 2 - 1      # (BxN_samples)xC
    
        if self.use_test_kwargs:               # return all outputs
            return rays_to_output(rgb), \
                   rays_to_output(disp), \
                   rays_to_output(acc), extras

        rgb = rays_to_output(rgb)
        return rgb

    def decrease_nerf_noise(self, it):
        end_it = 5000
        if it < end_it:
            noise_std = self.initial_raw_noise_std - self.initial_raw_noise_std/end_it * it
            self.render_kwargs_train['raw_noise_std'] = noise_std

    def sample_pose(self):   #計算旋轉矩陣(相機姿勢)  train
        # sample location on unit sphere
        #print("Type of self.v:", type(self.v))
        loc = sample_on_sphere(self.range_u, self.range_v)
        # loc = to_sphere(u, v)
        
        # sample radius if necessary
        radius = self.radius
        if isinstance(radius, tuple):
            radius = np.random.uniform(*radius)

        loc = loc * radius
        R = look_at(loc)[0]

        RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
        RT = torch.Tensor(RT.astype(np.float32))
        return RT

    def sample_select_pose(self, u, v):   #計算旋轉矩陣(相機姿勢)
        # sample location on unit sphere
        #print("Type of self.v:", type(self.v))
        
        # sample radius if necessary
        radius = self.radius
        if isinstance(radius, tuple):
            radius = np.random.uniform(*radius)
        
        if u == 0 and v == 0.5:  # 完全固定一個基準視角
            # 明確定義相機位置
            loc = np.array([radius, 0, 0])  # X軸正方向
            
            # 明確定義相機旋轉矩陣
            # 確保X軸指向相機右方，Y軸指向相機上方，Z軸指向相機後方
            R = np.array([
                [0, 0, 1],  # 右方向
                [1, 0, 0],  # 上方向 
                [0, 1, 0]  # 後方向（面向原點）
            ])
        else:
            # 正常的球面取樣
            loc = to_sphere(u, v) * radius
            R = look_at(loc)[0]
        
        RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
        RT = torch.Tensor(RT.astype(np.float32))
        
        return RT
    
    def sample_rays(self):   #設train用的rays
        pose = self.sample_pose()
        # print(f"`trainpose`:{pose}")
        sampler = self.val_ray_sampler if self.use_test_kwargs else self.ray_sampler 
        batch_rays, _, _ = sampler(self.H, self.W, self.focal, pose)
        return batch_rays #torch.Size([2, 1024, 3])
    
    def sample_select_rays(self, u ,v):
        pose = self.sample_select_pose(u, v)
        #print(f"trainpose:{pose}")
        sampler = self.val_ray_sampler if self.use_test_kwargs else self.ray_sampler  #如果 self.use_test_kwargs 為真，則使用 self.val_ray_sampler
        batch_rays, _, _ = sampler(self.H, self.W, self.focal, pose)
        return batch_rays

    def to(self, device):
        self.render_kwargs_train['network_fn'].to(device)
        self.device = device
        return self

    def train(self):
        self.use_test_kwargs = False
        self.render_kwargs_train['network_fn'].train()

    def eval(self):
        self.use_test_kwargs = True
        self.render_kwargs_train['network_fn'].eval()

    # def verify_camera_transform(self, c2w, expected_angle):
    #     # 將張量移到 CPU 並轉換為 NumPy
    #     if torch.is_tensor(c2w):
    #         camera_forward = c2w[:3, 2].detach().cpu().numpy()
    #     else:
    #         camera_forward = c2w[:3, 2]
        
    #     # 計算實際角度
    #     actual_angle = np.degrees(np.arctan2(camera_forward[1], camera_forward[0]))
    #     if actual_angle < 0:
    #         actual_angle += 360
            
    #     print(f"預期角度: {expected_angle}°")
    #     print(f"實際角度: {actual_angle:.2f}°")
    #     print(f"相機變換矩陣:\n{c2w.detach().cpu().numpy() if torch.is_tensor(c2w) else c2w}")