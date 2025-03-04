import numpy as np
import torch
from ..utils import sample_on_sphere, look_at, to_sphere
from ..transforms import FullRaySampler
from submodules.nerf_pytorch.run_nerf_mod import render, run_network            # import conditional render
from functools import partial
import torch.nn.functional as F  


class Generator(object):
    def __init__(self, H, W, focal, radius, ray_sampler, render_kwargs_train, render_kwargs_test, parameters, named_parameters,
                 range_u=(0,1), range_v=(0.01,0.49),v=0, chunk=None, device='cuda', orthographic=False, use_default_rays=False, reference_images=None, initial_direction=None):
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

        # 設置初始方向 (X 軸)
        if initial_direction is None:
            self.initial_direction = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        else:
            self.initial_direction = initial_direction
            
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

        # 初始化標準視角的射線
        canonical_pose = self.sample_select_pose(0, 0.5)  # 正面視角
        sampler = self.val_ray_sampler
        canonical_rays, _, _ = sampler(self.H, self.W, self.focal, canonical_pose)
        subsample_factor = 8
        subsampled_rays = canonical_rays[:, ::subsample_factor, :]
        self.canonical_rays = subsampled_rays
        
        self.reference_images = reference_images  # 字典，鍵為標籤，值為參考圖像
        self.consistency_weight = 0.5  # 一致性損失的權重

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
    
    # def fixed_coordinate_system(self, u, v, radius=1.0):
    #     """
    #     创建一个固定的坐标系统，确保每次调用时结果都相同
        
    #     参数:
    #         u: 水平参数 (0-1)，控制绕Z轴的旋转
    #         v: 垂直参数 (0-1)，控制与Z轴的夹角
    #         radius: 距离原点的距离
            
    #     返回:
    #         RT: 4x4变换矩阵的前3行，格式为torch.Tensor
    #     """
    #     # 将u,v转换为球坐标
    #     theta = 2 * np.pi * u     # 水平角度 (0-2π)
    #     phi = np.arccos(1 - 2 * v)  # 垂直角度 (0-π)
        
    #     # 计算相机位置
    #     x = radius * np.sin(phi) * np.cos(theta)
    #     y = radius * np.sin(phi) * np.sin(theta)
    #     z = radius * np.cos(phi)
    #     position = np.array([x, y, z])
        
    #     # 根据原始代码的定义计算视图矩阵
    #     # 从原始look_at函数的实现，我们知道:
    #     # z_axis是从相机指向目标(这里是原点)的方向
    #     # 全局上方向为[0, 0, 1]
        
    #     # 1. 计算z轴方向 (从相机指向原点)
    #     z_axis = position  # 相机看向原点
    #     z_axis = z_axis / np.linalg.norm(z_axis)
        
    #     # 2. 定义全局上方向
    #     up = np.array([0, 0, 1])
        
    #     # 3. 计算x轴方向 (右方向)
    #     x_axis = np.cross(up, z_axis)
    #     if np.linalg.norm(x_axis) < 1e-6:
    #         # 如果相机在Z轴上，使用固定的右方向
    #         x_axis = np.array([1, 0, 0])
    #     else:
    #         x_axis = x_axis / np.linalg.norm(x_axis)
        
    #     # 4. 计算y轴方向 (上方向)
    #     y_axis = np.cross(z_axis, x_axis)
    #     y_axis = y_axis / np.linalg.norm(y_axis)
        
    #     # 5. 构建旋转矩阵
    #     # 根据原始look_at函数，旋转矩阵由x_axis, y_axis, z_axis组成
    #     r_mat = np.stack([x_axis, y_axis, z_axis], axis=1)
        
    #     # 6. 构建变换矩阵，将位置向量添加为最后一列
    #     RT = np.concatenate([r_mat, position.reshape(3, 1)], axis=1)
        
    #     # 7. 转换为torch.Tensor
    #     return torch.tensor(RT, dtype=torch.float32)
    def fixed_world_coordinate_system(self, u, v, radius=1.0):
        """
        創建一個固定的世界座標系統，其中 X 軸與訓練模型的起始點對齊
        
        參數:
            angle_degrees: 水平角度 (0-360 度)，從 X 軸正方向開始
            elevation_degrees: 仰角 (0-90 度)，從 XY 平面向上測量
            radius: 相機距離原點的距離
        """
        initial_dir = self.initial_direction.cpu().numpy()
        initial_dir = initial_dir / np.linalg.norm(initial_dir)

        # 將角度轉換為弧度
        theta = 2 * np.pi * u
        phi = np.arccos(1 - 2 * v)

        if np.allclose(initial_dir, [1, 0, 0]):
            # 如果初始方向已經是 X 軸，不需要額外旋轉
            align_matrix = np.eye(3)
        else:
            # 計算旋轉軸（初始方向與 X 軸的叉積）
            rotation_axis = np.cross([1, 0, 0], initial_dir)
            if np.linalg.norm(rotation_axis) < 1e-5:
                # 如果初始方向與 X 軸平行但方向相反
                align_matrix = np.diag([-1, -1, 1])  # 繞 Z 軸旋轉 180 度
            else:
                # 標準化旋轉軸
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                # 計算旋轉角度（初始方向與 X 軸的夾角）
                cos_angle = np.dot(initial_dir, [1, 0, 0])
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                # 使用羅德里格斯旋轉公式計算旋轉矩陣
                K = np.array([
                    [0, -rotation_axis[2], rotation_axis[1]],
                    [rotation_axis[2], 0, -rotation_axis[0]],
                    [-rotation_axis[1], rotation_axis[0], 0]
                ])
                align_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
        
        # 計算相機在球面上的位置
        # 這裡 X 軸是您模型的起始點方向
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        
        # 相機位置向量
        pos_vector = np.array([x, y, z])
        
        # 使用對齊矩陣調整相機位置，使其相對於初始方向
        camera_pos = align_matrix @ pos_vector
        
        # 視線方向 - 指向原點
        view_dir = camera_pos / np.linalg.norm(camera_pos)
        
        # 上方向 - 固定為全局 Z 軸
        up = np.array([0, 0, 1])
        
        # 計算相機坐標系中的 x 軸 (右方向)
        x_axis = np.cross(up, view_dir)
        if np.linalg.norm(x_axis) < 1e-5:
            # 如果相機位於 Z 軸上，使用固定的 X 軸
            if z > 0:  # 在 Z 軸上方
                x_axis = np.array([1, 0, 0])
            else:      # 在 Z 軸下方
                x_axis = np.array([-1, 0, 0])
        else:
            x_axis = x_axis / np.linalg.norm(x_axis)
        
        # 計算相機坐標系中的 y 軸 (上方向)
        y_axis = np.cross(view_dir, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # z 軸是視線方向
        z_axis = view_dir
        
        # 構建旋轉矩陣
        r_mat = np.stack([x_axis, y_axis, z_axis], axis=1)
        
        # 構建變換矩陣
        RT = np.concatenate([r_mat, camera_pos.reshape(3, 1)], axis=1)
        
        return torch.tensor(RT, dtype=torch.float32)

    def sample_select_pose(self, u, v):   #計算旋轉矩陣(相機姿勢)
        # sample location on unit sphere
        #print("Type of self.v:", type(self.v))
        
        # sample radius if necessary
        radius = self.radius
        # if isinstance(radius, tuple):
        #     radius = np.random.uniform(*radius)
        
        # if u == 0 and v == 0.5:  # 完全固定一個基準視角
        #     # 明確定義相機位置
        #     loc = np.array([radius, 0, 0])  # X軸正方向
            
        #     # 明確定義相機旋轉矩陣
        #     # 確保X軸指向相機右方，Y軸指向相機上方，Z軸指向相機後方
        #     R = np.array([
        #         [0, 0, 1],  # 右方向
        #         [1, 0, 0],  # 上方向 
        #         [0, 1, 0]  # 後方向（面向原點）
        #     ])
        # else:
        #     # 正常的球面取樣
        #     loc = to_sphere(u, v) * radius
        #     R = look_at(loc)[0]
        
        # RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
        # RT = torch.Tensor(RT.astype(np.float32))
        RT = self.fixed_world_coordinate_system(u, v, radius)
        
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

    def compute_consistency_loss(self, z, label):
        """單獨計算一致性損失的方法"""
        if self.reference_images is None:
            return 0.0
            
        with torch.cuda.amp.autocast(enabled=True):
            consistency_loss = 0
            consistency_count = 0
            
            for i, l in enumerate(label):
                pillar_type = int(l[0].item())
                
                if pillar_type in self.reference_images:
                    ref_label = torch.tensor([[pillar_type, 0, 0]]).to(label.device)
                    
                    # 從固定視角渲染
                    subsample_factor = 1  # 採樣率降低4倍
                    subsampled_rays = self.canonical_rays[:, ::subsample_factor, :]
                    render_kwargs = self.render_kwargs_train.copy()
                    render_kwargs['features'] = z[i:i+1]
                    
                    ref_rgb, _, _, _ = render(self.H//subsample_factor, self.W//subsample_factor, 
                            self.focal/subsample_factor,
                            ref_label, chunk=self.chunk, rays=subsampled_rays,
                            **render_kwargs)
                    ref_rgb = ref_rgb.view(len(ref_rgb), -1) * 2 - 1
                    
                    # 比較參考圖像
                    ref_img = self.reference_images[pillar_type].to(ref_rgb.device)
                    consistency_loss += F.mse_loss(ref_rgb, ref_img)
                    consistency_count += 1
            
            if consistency_count > 0:
                return consistency_loss / consistency_count
            return 0.0
        
    