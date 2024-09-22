import os
import imageio
import torch.nn as nn
import time
import torch
import torch.nn.functional as F
import numpy as np
from .mlp import MLP
from .utils import *
from lib.config import cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False

def batchify(fn, chunk):
    """
    Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    if isinstance(chunk, str):
        try:
            chunk = int(chunk)
        except ValueError:
            raise ValueError(f"Invalid value for cfg.task_arg.netchunk: {chunk}. It should be an integer.")

    def ret(inputs):
        # 以chunk分批进入网络，防止显存爆掉，然后在拼接
        # FIXME:这里不知道为什么参数传过来就变str了，导致bug
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """
    被下面的create_nerf 封装到了lambda方法里面
    Prepares inputs and applies network 'fn'.
    inputs: pts，光线上的点 如 [1024,64,3]，1024条光线，一条光线上64个点
    viewdirs: 光线起点的方向
    fn: 神经网络模型 粗糙网络或者精细网络
    embed_fn:
    embeddirs_fn:
    netchunk:
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])  # [N_rand*64,3]
    # 坐标点进行编码嵌入 [N_rand*64,63]
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        # 方向进行位置编码
        embedded_dirs = embeddirs_fn(input_dirs_flat)  # [N_rand*64,27]
        embedded = torch.cat([embedded, embedded_dirs], -1)

    # 里面经过网络 [bs*64,3]
    outputs_flat = batchify(fn, netchunk)(embedded)
    # [bs*64,4] -> [bs,64,4]
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


class Network(nn.Module):
    def __init__(self,):

        super(Network, self).__init__()

        self._init_pipeline(cfg)

    def _init_pipeline(self, cfg):
        net_cfg = cfg.network
        embed_fn, input_ch = get_embedder(net_cfg.multires, net_cfg.i_embed)
        input_ch_views = 0
        embeddirs_fn = None
        if net_cfg.use_viewdirs:
            embeddirs_fn, input_ch_views = get_embedder(net_cfg.multires_views, net_cfg.i_embed)

        # 想要=5生效，首先需要use_viewdirs=False and N_importance>0
        output_ch = 5 if net_cfg.nerf.N_importance > 0 else 4
        skips = [4]
        # 粗网络
        self.model = MLP(D=net_cfg.nerf.D, W=net_cfg.nerf.W,
                     input_ch=input_ch, output_ch=output_ch, skips=skips,
                     input_ch_views=input_ch_views, use_viewdirs=net_cfg.use_viewdirs).to(device)
        self.model_fine = None
        if net_cfg.nerf.N_importance > 0:
            # 精细网络
            self.model_fine = MLP(D=net_cfg.nerf.D_fine, W=net_cfg.nerf.W_fine,
                              input_ch=input_ch, output_ch=output_ch, skips=skips,
                              input_ch_views=input_ch_views, use_viewdirs=net_cfg.use_viewdirs).to(device)

        # netchunk 是网络中处理的点的batch_size
        self.network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                            embed_fn=embed_fn,
                                                                            embeddirs_fn=embeddirs_fn,
                                                                            netchunk=cfg.task_arg.netchunk)

        # Prepare ray batch tensor if batching random rays
        self.N_rand = cfg.task_arg.N_rays
        self.use_batching = not cfg.task_arg.no_batching
        self.chunk = cfg.task_arg.chunk_size

        self.render_kwargs_train = {
        'network_query_fn': self.network_query_fn,
        'perturb': net_cfg.perturb,
        # 精细网络每束光线上的采样点数量
        'N_importance': net_cfg.nerf.N_importance,
        # 精细网络
        'network_fine': self.model_fine,
        # 粗网络每束光线上的采样点数量
        'N_samples': net_cfg.nerf.N_samples,
        # 粗网络
        'network_fn': self.model,
        'use_viewdirs': net_cfg.use_viewdirs,
        'white_bkgd': cfg.task_arg.white_bkgd,
        'raw_noise_std': net_cfg.raw_noise_std
        }

        print(self.model_fine)
        # NDC only good for LLFF-style forward facing data
        if cfg.scene != 'llff' or net_cfg.no_ndc:
            print('Not ndc!')
            self.render_kwargs_train['ndc'] = False
            self.render_kwargs_train['lindisp'] = net_cfg.lindisp
        self.render_kwargs_test = {k: self.render_kwargs_train[k] for k in self.render_kwargs_train}
        self.render_kwargs_test['perturb'] = False
        self.render_kwargs_test['raw_noise_std'] = 0.
        bds_dict = {
            'near': net_cfg.near,
            'far': net_cfg.far,
        }
        self.render_kwargs_train.update(bds_dict)
        self.render_kwargs_test.update(bds_dict)

    def render(self, H, W, K, chunk=1024 * 32, rays=None, c2w=None, ndc=True,
               near=0., far=1., use_viewdirs=False, c2w_staticcam=None, **kwargs):
        """Render rays
        Args:
          H: int. Height of image in pixels.
          W: int. Width of image in pixels.
          K:  相机内参
          chunk: int. Maximum number of rays to process simultaneously. Used to
            control maximum memory usage. Does not affect final results.
          rays: array of shape [2, batch_size, 3]. Ray origin and direction for
            each example in batch.
          c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
          ndc: bool. If True, represent ray origin, direction in NDC coordinates.
          near: float or array of shape [batch_size]. Nearest distance for a ray.
          far: float or array of shape [batch_size]. Farthest distance for a ray.
          use_viewdirs: bool. If True, use viewing direction of a point in space in model.
          c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
           camera while using other c2w argument for viewing directions.
        Returns:
          rgb_map: [batch_size, 3]. Predicted RGB values for rays.
          disp_map: [batch_size]. Disparity map. Inverse of depth.
          acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
          extras: dict with everything returned by render_rays().
        """

        if c2w is not None:
            # special case to render full image
            """
            得到的是o是相机在世界系下的坐标，d也是世界系下的方向向量
            rays_d [400,400,3]
            rays_o [400,400,3]
            """
            rays_o, rays_d = get_rays(H, W, K, c2w)
        else:
            # use provided ray batch
            # 光线的起始位置, 方向
            rays_o, rays_d = rays

        if use_viewdirs:
            # provide ray directions as input
            viewdirs = rays_d
            # 静态相机 相机坐标到世界坐标的转换
            if c2w_staticcam is not None:
                # special case to visualize effect of viewdirs
                rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
            # 单位向量 [HW,3]
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        # rays_d [400,400,3](也许不是这个形状 当condition不同时)
        sh = rays_d.shape  # [..., 3]

        if ndc:
            # for forward facing scenes
            rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

        # Create ray batch
        # (HW, 3)
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()
        # run_render_only的时候的形状： [HW,1],[HW,1]
        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
        # 8=3+3+1+1 把世界系下的相机原点（x,y,z）射线向量(x',y',z')和view frustum的near和far concat起来
        rays = torch.cat([rays_o, rays_d, near, far], -1)
        if use_viewdirs:  # viewdirs其实就是rays_d的归一化方向向量，在这又cat了一遍
            # 加了direction的三个坐标
            # 3 3 1 1 3
            rays = torch.cat([rays, viewdirs], -1)  # [bs,11]

        # Render and reshape

        # rgb_map,disp_map,acc_map,raw,rbg0,disp0,acc0,z_std
        all_ret = self.batchify_rays(rays, chunk, **kwargs)
        for k in all_ret:
            # 对所有的返回值进行reshape
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        # 讲精细网络的输出单独拿了出来
        k_extract = ['rgb_map', 'disp_map', 'acc_map']
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
        # 前三是list，后5还是在map中
        return ret_list + [ret_dict]

    def batchify_rays(self, rays_flat, chunk=1024 * 32, **kwargs):
        """
        Render rays in smaller minibatches to avoid OOM.
        rays_flat: [N_rand,11]
        """
        all_ret = {}
        for i in range(0, rays_flat.shape[0], chunk):
            ret = self.render_rays(rays_flat[i:i + chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        # 将分批处理的结果拼接在一起
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret

    def render_rays(self, ray_batch,
                    network_fn,
                    network_query_fn,
                    N_samples,
                    retraw=False,
                    lindisp=False,
                    perturb=0.,
                    N_importance=0,
                    network_fine=None,
                    white_bkgd=False,
                    raw_noise_std=0.,
                    verbose=False,
                    pytest=False):
        """Volumetric rendering.
        Args:
          ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction. 单位大小查看方向
          粗网络
          network_fn: function. Model for predicting RGB and density at each point
            in space.
          network_query_fn: function used for passing queries to network_fn.
          N_samples: int. Number of different times to sample along each ray.

          raw 是指神经网络的输出
          retraw: bool. If True, include model's raw, unprocessed predictions.
          lindisp: bool. If True, sample linearly in inverse depth rather than in depth.


          perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified random points in time.

          精细网络中的光线上的采样频率
          N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
          精细网络
          network_fine: "fine" network with same spec as network_fn.
          white_bkgd: bool. If True, assume a white background. 白色背景
          raw_noise_std: ...


          verbose: bool. If True, print more debugging info.
        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
          disp_map: [num_rays]. Disparity map. 1 / depth.
          acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
          raw: [num_rays, num_samples, 4]. Raw predictions from model.
          rgb0: See rgb_map. Output for coarse model.
          disp0: See disp_map. Output for coarse model.
          acc0: See acc_map. Output for coarse model.
          z_std: [num_rays]. Standard deviation of distances along ray for each
            sample.
        """
        # 从batch中分出来的1024*32大小的mini batch避免显存占用过大
        N_rays = ray_batch.shape[0]  # N_rand
        # 光线起始位置，光线的方向
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        # 视角的单位向量
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])  # [bs,1,2] near和far
        near, far = bounds[..., 0], bounds[..., 1]  # [bs,1]
        # 采样点
        t_vals = torch.linspace(0., 1., steps=N_samples).to(near.device)
        if not lindisp:
            # FIXME: not in same device bug
            z_vals = near * (1. - t_vals) + far * (t_vals) # 插值采样
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
        # [N_rand,64] -> [N_rand,64]
        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            # get intervals between samples，64个采样点的中点
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper.device)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand)
            # [bs,64] 加上随机的噪声
            z_vals = lower + (upper - lower) * t_rand

        # 空间中的采样点
        # [N_rand, 64, 3]
        # 出发点+距离*方向
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

        # 使用神经网络 viewdirs [N_rand,3], network_fn 指的是粗糙NeRF或者精细NeRF
        # raw [bs,64,3+1]
        raw = network_query_fn(pts, viewdirs, network_fn)

        # rgb值，xx，权重的和，weights就是论文中的那个Ti和alpha的乘积
        rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                     pytest=pytest)

        # 精细网络部分
        if N_importance > 0:
            # _0 是第一个阶段 粗糙网络的结果
            # 这三个留着放在dict中输出用
            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
            # 第二次计算mid，取中点位置
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            # 给精细网络使用的点
            # [N_rays, N_samples + N_importance, 3]
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

            run_fn = network_fn if network_fine is None else network_fine

            # 使用神经网络
            # create_nerf 中的 network_query_fn 那个lambda 函数
            # viewdirs 与粗糙网络是相同的
            raw = network_query_fn(pts, viewdirs, run_fn)

            rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                         pytest=pytest)

        ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}

        if retraw:
            # 如果是两个网络，那么这个raw就是最后精细网络的输出
            ret['raw'] = raw

        if N_importance > 0:
            # 下面的0是粗糙网络的输出
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0

            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        # 检查是否有异常值
        for k in ret:
            if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
                print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret

    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            Model的输出
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            采样，并未变成空间点的那个采样点
            z_vals: [num_rays, num_samples along ray]. Integration time.
            光线的方向
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            RGB颜色值
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            逆深度
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            权重和？
            acc_map: [num_rays]. Sum of weights along each ray.
            权重
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            估计的深度
            depth_map: [num_rays]. Estimated distance to object.
        """
        # Alpha的计算
        # relu, 负数拉平为0
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)
        # [bs,63]
        # 采样点之间的距离
        """
        dists[..., :1]和dists[..., 0]的区别就是一个会保留这个切片维度，
        一个不保留，所以out分别是torch.Size([32768, 1])和torch.Size([32768])
        """
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        # TODO: 为什么这块要给最后一个特别大的数？
        dists = torch.cat([dists, torch.tensor([1e10],device=dists.device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
        # rays_d[...,None,:] [bs,3] -> [bs,1,3]
        # 1维 -> 3维
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        # RGB经过sigmoid处理
        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[..., 3].shape) * raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
                noise = torch.Tensor(noise)
        # 计算公式3 [bs, 64],
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

        # 后面这部分就是Ti，前面是alpha，这个就是论文上的那个权重w [bs,64]
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device),
                                                   1. - alpha + 1e-10], -1),
                                        -1)[:, :-1]
        # [bs, 64,1] * [bs,64,3]
        # 在第二个维度，64将所有的点的值相加 -> [32,3]
        # 公式3的结果值
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        # (32,)
        # 深度图
        # Estimated depth map is expected distance.
        depth_map = torch.sum(weights * z_vals, -1)
        # 视差图
        # Disparity map is inverse depth.
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))

        # 权重和
        # 这个值仅做了输出用，后续并无使用
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map

    def forward(self, inputs):
        # TODO: maybe more elegant here
        if inputs['meta']['is_train'][0] == True:
            # Cast intrinsics to right types
            H, W, focal = inputs['hwf']
            # TODO: avoid hardcode, 所有的HWF对于同一数据集应该相同，想更高效的传参方法
            K = inputs['K'][0]
            H, W, focal= int(H[0]), int(W[0]), int(focal[0])
            # 随机选出N_rand个光线作为一个mini batch: N_rays, 3, 3
            if self.use_batching:
                # b, N_rays, 3, 3
                batch = inputs['rays_rgb'][:, :self.N_rand].flatten(0, 1)
                # 3, N_rays, 3
                batch = torch.transpose(batch, 0, 1)
                # batch_rays:(2, N_rays, 3), target_s:(N_rays, 3): label应该不需要额外通道1
                batch_rays, target_s = batch[:2], batch[2]  # 前两个是rays_o和rays_d, 第三个是target就是image的rgb
            else:
                # FIXME: set this condition in the future after finish baseline in use_batching condition
                image = inputs['img']
                pose = inputs['pose']
                rays_o, rays_d = get_rays(H, W, K, pose) # (H, W, 3), (H, W, 3)
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H),
                                                    torch.linspace(0, W - 1, W), indexing='ij'),
                                     -1)  # (H, W, 2)
                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                # 生成一个随机的索引序列
                select_inds = torch.randperm(coords.shape[0])[:self.N_rand]
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)  # 堆叠 o和d
                # target 也同样选出对应位置的点
                # target 用来最后的mse loss 计算
                target_s = image[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            # rgb 网络计算出的图像
            # 前三是精细网络的输出内容，其他的还保存在一个dict中，有5项
            rgb, disp, acc, extras = self.render(H, W, K, chunk=self.chunk, rays=batch_rays,
                                            verbose=False, retraw=True,
                                            **self.render_kwargs_train)

        else:
            H, W, focal = inputs['hwf']
            K = inputs['K'][0]
            H, W, focal = int(H[0]), int(W[0]), int(focal[0])
            rgb, disp, acc, extras = self.render(H, W, K, chunk=self.chunk, c2w=inputs['pose'][0, :3, :4], **self.render_kwargs_test)
            target_s = inputs['img'][0]
        ret = {}
        # 精细网络输出
        ret['rgb'] = rgb
        ret['disp'] = disp
        ret['acc'] = acc
        # 其余输出的list(dict)
        ret['extras'] = extras
        # gt
        ret['target_s'] = target_s

        return ret