'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-08-14 19:47:31
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-08-15 12:11:45
FilePath: /ArSSR-org/model/Hloss_FF.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch

class HDRLoss_FF(torch.nn.Module):
    """
    HDR loss function with frequency filtering (v4)
    """
    def __init__(self, hdr_ff_sigma=1, hdr_eps=1e-2, hdr_ff_factor=0):
        super().__init__()
        self.sigma = float(hdr_ff_sigma)
        self.eps = float(hdr_eps)
        self.factor = float(hdr_ff_factor)

    def forward(self, input, target, kcoords, weights=None, reduce=True):
        # target_max = target.abs().max()
        # target /= target_max
        # input = input / target_max
        # input_nograd = input.clone()
        # input_nograd = input_nograd.detach()
        dist_to_center2 = kcoords[..., 0]**2 + kcoords[..., 1]**2 + kcoords[..., 2]**2
        filter_value = torch.exp(-dist_to_center2/(2*self.sigma**2)).unsqueeze(-1)

        if input.dtype == torch.float:
            input = torch.view_as_complex(input) #* filter_value
        if target.dtype == torch.float:
            target = torch.view_as_complex(target)
        
        assert input.shape == target.shape
        target = torch.log(abs(target) + 1e-4)
        input = torch.log(abs(input) + 1e-4)
        error = input - target
        # error = error * filter_value

        loss = (error.abs()/(input.detach().abs()+self.eps))**2
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)

        reg_error = (input - input * filter_value)
        reg = self.factor * (reg_error.abs()/(input.detach().abs()+self.eps))**2
        # reg = torch.matmul(torch.conj(reg).t(), reg)
        # reg = reg.abs() * self.factor
        # reg = torch.zeros([1]).mean()

        if reduce:
            return loss.mean() + reg.mean(), reg.mean()
        else:
            return loss, reg