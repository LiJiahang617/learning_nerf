import torch
import torch.nn as nn
from lib.utils import net_utils
from lib.config import cfg

class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader):
        super(NetworkWrapper, self).__init__()
        self.net = net
        # 为了实现per-sample的loss计算，在这里保持loss形状与输入相同，不reduct
        self.color_crit = nn.MSELoss(reduction='mean')
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.tensor([10.], device=x.device))

    def forward(self, batch):
        output = self.net(batch)
        batch_size = batch['img'].shape[0]
        scalar_stats = {}
        loss = 0
        rgb = output['rgb'].view(batch_size, -1, 3)
        target_s = output['target_s'].view(batch_size, -1, 3)
        color_loss = self.color_crit(rgb, target_s)
        # detach()操作会生成一个新的张量，与原始张量共享数据，不会修改原始张量
        scalar_stats.update({'color_mse': color_loss})
        loss += color_loss
        # 计算PSNR
        psnr = -10. * torch.log(color_loss.detach()) / \
               torch.log(torch.Tensor([10.]).to(color_loss.device))
        scalar_stats.update({'psnr': psnr})

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats
