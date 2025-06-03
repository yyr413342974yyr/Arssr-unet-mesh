# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: utils.py
#   Author  : Qing Wu
#   Email   : wuqing@shanghaitech.edu.cn
#   Date    : 2021/9/19
# -----------------------------------------
import os
import torch
import numpy as np
import SimpleITK as sitk
from skimage.metrics import structural_similarity
from tqdm import tqdm
import nibabel as nib

def read_img(in_path):
    img_list = []

    # 检查传入的路径是文件还是目录
    if os.path.isdir(in_path):
        # 是目录，遍历目录中的所有文件
        filenames = os.listdir(in_path)
        for f in tqdm(filenames):
            if f.endswith('.nii') or f.endswith('.nii.gz'):
                img = nib.load(os.path.join(in_path, f))
                img_vol = img.get_fdata()
                img_list.append(img_vol)
    elif os.path.isfile(in_path):
        # 是文件，直接读取文件
        if in_path.endswith('.nii') or in_path.endswith('.nii.gz'):
            img = nib.load(in_path)
            img_vol = img.get_fdata()
            img_list = img_vol
    else:
        raise ValueError("Provided path is neither a directory nor a file.")

    return img_list


# -------------------------------
# here coder is from https://github.com/yinboc/liif/blob/main/utils.py
# -------------------------------
def make_coord(shape, ranges=None, flatten=True):
    """
    Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def psnr(image, ground_truth):
    mse = np.mean((image - ground_truth) ** 2)
    if mse == 0.:
        return float('inf')
    data_range = np.max(ground_truth) - np.min(ground_truth)
    return 20 * np.log10(data_range) - 10 * np.log10(mse)


def ssim(image, ground_truth):
    data_range = np.max(ground_truth) - np.min(ground_truth)
    return structural_similarity(image, ground_truth, data_range=data_range)

def compute_nmse(true_img, pred_img):
    # 初始化用于存储每个通道NMSE的数组
    nmse_per_channel = np.zeros(true_img.shape[-1])
    # 对每个颜色通道计算NMSE
    for i in range(true_img.shape[-1]):
        # 提取单个通道的图像
        true_img_channel = true_img[...,i]
        pred_img_channel = pred_img[...,i]
        # 计算MSE
        mse = np.mean((true_img_channel - pred_img_channel) ** 2)
        # 计算信号能量
        signal_energy = np.mean(true_img_channel ** 2)
        # 计算此通道的NMSE并存储
        nmse_per_channel[i] = mse / signal_energy
    # 计算所有通道的平均NMSE
    nmse = np.mean(nmse_per_channel)
    return nmse

def write_img(vol, out_path, ref_path, new_spacing=None):
    ref_img = nib.load(ref_path)
    affine = ref_img.affine

    if new_spacing is not None:
        # If new spacing is provided, adjust the affine matrix accordingly
        zoom_factors = np.array(ref_img.header.get_zooms()[:3]) / np.array(new_spacing)
        new_affine = np.copy(affine)
        for i in range(3):
            new_affine[i, i] /= zoom_factors[i]
    else:
        new_affine = affine

    new_img = nib.Nifti1Image(vol, new_affine)
    nib.save(new_img, out_path)
    print('Saved to:', out_path)

def normal(in_image):
    value_max = np.max(in_image)
    value_min = np.min(in_image)
    # 添加一个小的常数 epsilon 防止除零错误
    epsilon = 1e-10
    return (in_image - value_min) / (value_max - value_min + epsilon)



