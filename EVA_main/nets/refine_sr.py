from copy import deepcopy
import torch
from torch import nn as nn
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F
import math
import os
import time

def reg_dense_conf(x, mode):
    """
    extract confidence from prediction head output
    """
    mode, vmin, vmax = mode
    if mode == 'exp':
        return vmin + x.exp().clip(max=vmax-vmin)
    if mode == 'sigmoid':
        return (vmax - vmin) * torch.sigmoid(x) + vmin
    raise ValueError(f'bad {mode=}')


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)



class ResidualDenseBlock_Conf(nn.Module):
    """Residual Dense Block.
    Used in RRDB block in ESRGAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64):
        super(ResidualDenseBlock_Conf, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, 3, 1, 1, 0)
        self.conv2 = nn.Conv2d(num_feat + 3, 1, 1, 1, 0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # initialization
        default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.conv2(torch.cat((x, x1), 1))
        x2 = reg_dense_conf(x2, mode=('exp', 1, float('inf'))) / 2.0
        return x2



if __name__ == '__main__':
    net = ResidualDenseBlock_Conf(4)
    inp = torch.zeros(1, 3, 1200, 800)
    inp2 = torch.zeros(1, 1, 1200, 800)
    inp_all = torch.cat([inp, inp2], dim=1)
    out = net(inp_all)
    print(out.shape)