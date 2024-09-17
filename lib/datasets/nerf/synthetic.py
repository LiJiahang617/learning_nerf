import torch
import torch.utils.data as data
import numpy as np
import os
from lib.utils import data_utils
from lib.config import cfg
from torchvision import transforms as T
import imageio
import json
import cv2
from lib.networks.nerf.utils import get_rays_np


class Dataset(data.Dataset):
    def __init__(self, half_res=False, **kwargs):
        super(Dataset, self).__init__()
        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene
        input_ratio = kwargs['input_ratio']
        cams = kwargs['cams']
        self.half_res = half_res

        self.input_ratio = kwargs['input_ratio']
        self.data_root = os.path.join(data_root, scene)
        self.split = split
        # number of random rays per training iteration
        self.N_rays = cfg.task_arg.N_rays
        # chunkify, number of rays processed in parallel, decrease if running out of memory
        # FIXME: may have no use in this framework
        # self.chunk = cfg.task_arg.chunk_size
        # use white background
        self.white_bkgd = cfg.task_arg.white_bkgd
        # importance sampling, TODO: next two items maybe not needed here?
        self.N_samples = cfg.network.nerf.N_samples
        self.N_importance = cfg.network.nerf.N_importance
        self.use_batching = not cfg.task_arg.no_batching

        # read image and pose
        self.images = []
        self.poses = []
        self.rotation = []
        json_info = json.load(open(os.path.join(self.data_root, 'transforms_{}.json'.format(split)), 'r'))
        self.camera_angle_x = json_info['camera_angle_x']
        for frame in json_info['frames']:
            fname = os.path.join(self.data_root, frame['file_path'][2:] + '.png')
            self.images.append(fname)
            self.poses.append(np.array(frame['transform_matrix']))
            self.rotation.append(float(frame['rotation']))

        # 平移
        self.trans_t = lambda t: torch.Tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1]]).float()

        # 绕x轴的旋转
        self.rot_phi = lambda phi: torch.Tensor([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1]]).float()

        # 绕y轴的旋转
        self.rot_theta = lambda th: torch.Tensor([
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1]]).float()

    def pose_spherical(self, theta, phi, radius):
        """
        theta: -180 -- +180，间隔为9
        phi: 固定值 -30
        radius: 固定值 4
        """
        c2w = self.trans_t(radius)
        c2w = self.rot_phi(phi / 180. * np.pi) @ c2w
        c2w = self.rot_theta(theta / 180. * np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
        return c2w

    def __getitem__(self, index):
        img_path = self.images[index]
        img = imageio.imread(img_path)
        # (4, 4)
        pose = self.poses[index]

        # 归一化 imgs: [num_split_images * (H, W, 4)] --> array: (num_split_images, H, W, 4)
        img = (np.array(img) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)，4通道 rgba
        # poses: [num_split_images * (4, 4)] --> array: (num_split_images, 4, 4)
        pose = np.array(pose).astype(np.float32)
        H, W = img.shape[:2]
        # camera_angle_x代表水平FOV 根据这个可以算出焦距f, 焦距 公式正确
        focal = .5 * W / np.tan(.5 * self.camera_angle_x)
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])
        if self.half_res:
            H = H // 2
            W = W // 2
            # 焦距一半
            focal = focal / 2.

            # 调整成一半的大小
            img_half_res = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            img = img_half_res
        #  np.linspace(-180, 180, 40 + 1) 9度一个间隔
        # (40,4,4), 渲染的结果就是40帧
        render_poses = torch.stack(
            [self.pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)
        if self.white_bkgd:
            img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:])
        else:
            # (H, W, 3)
            img = img[..., :3]
        pose = pose[:3, :4]
        # [2(ro+rd), H, W, 3]
        rays = np.stack(get_rays_np(H, W, K, pose), 0)
        # [3(ro+rd+rgb), H, W, 3]
        rays_rgb = np.concatenate([rays, img[None, ...]], 0)
        # [H, W, 3(ro+rd+rgb), 3]
        rays_rgb = np.transpose(rays_rgb, [1, 2, 0, 3])
        # # [H*W, 3(ro+rd+rgb), 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        # 打乱光线
        np.random.shuffle(rays_rgb)
        # 这个框架后面会在trainer统一to cuda，这里只需要做好一个样本tenor就行
        img = torch.Tensor(img)
        rays_rgb = torch.Tensor(rays_rgb)
        # # 随机选出N_rand个光线作为一个mini batch: N_rays, 3, 3
        # batch = rays_rgb[:self.N_rays]
        # # 3, N_rays, 3
        # batch = torch.transpose(batch, 0, 1)
        # # batch_rays:(2, N_rays, 3), target_s:(N_rays, 3): label应该不需要额外通道1
        # batch_rays, target_s = batch[:2], batch[2]  # 前两个是rays_o和rays_d, 第三个是target就是image的rgb
        # TODO: finish load data Random from one image if not use_batching
        """
        K在一开始算rays（use_batching条件下）的时候需要（nerf官方代码中），后续几次调用K都是在处理特殊情况下才会需要：
        1. 不用use_batching条件下用来算rays
        2. 给了c2w时用来算rays_o/rays_d
        3. 给了c2w_staticcam时用来算rays_o/rays_d
        4. 开了ndc情况下计算新的rays_o, rays_d
        其他情况即使函数传了K，也不会用到
        目前先试试默认传一个K
        """
        ret = {'img': img, 'pose': pose, 'hwf': [H, W, focal], 'rays_rgb': rays_rgb, 'K': K,
               'render_poses': render_poses} # input and output. they will be sent to cuda

        return ret

    def __len__(self):

        return len(self.images)
