# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: test.py
#   Author  : Qing Wu
#   Email   : wuqing@shanghaitech.edu.cn
#   Date    : 2021/9/19
# -----------------------------------------
import SimpleITK
import numpy as np
import os
from model import model
import utils
import torch
import argparse
from tqdm import tqdm
from utils import normal
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from data import data
import nibabel as nib
import random
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

    # -----------------------
    # parameters settings
    # -----------------------
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', type=str, default='ResCNN_MLP_PE_normal', dest='experiment_name',
                        help='experiment_name')

    # about ArSSR model
    parser.add_argument('--encoder', type=str, default='ResCNN', dest='encoder_name',
                        help='the type of encoder network, including RDN (default), ResCNN, and SRResnet.')
    parser.add_argument('-decoder_name', type=str, default='MLP', dest='decoder_name',
                        help='the type of decoder network, including MLP, Siren, Siren_res') 
    parser.add_argument('-decoder_depth', type=int, default=8, dest='decoder_depth',
                        help='the depth of the decoder network (default=8).')
    parser.add_argument('-decoder_width', type=int, default=256, dest='decoder_width',
                        help='the width of the decoder network (default=256).')
    parser.add_argument('-feature_dim', type=int, default=128, dest='feature_dim',
                        help='the dimension size of the feature vector (default=128)')
    parser.add_argument('-PE_embedding', type=str, default="True", dest='PE_embedding',
                        help='the dimension size of the feature vector (default=128)')
    parser.add_argument('-freq', type=int, default= 9 , dest='freq', help='')
                       

    # about GPU
    parser.add_argument('--is_gpu', type=int, default=1, dest='is_gpu', help='enable GPU (1->enable, 0->disenable)')
    parser.add_argument('--gpu', type=int, default=0, dest='gpu', help='the number of GPU')
                      
    # about file
    parser.add_argument('--input_path', type=str, default='/data3/langzhang/meshfitting/DATA/x2/', dest='input_path',
                        help='the file path of LR input image')
    parser.add_argument('--ref_path', type=str, default='/data3/langzhang/meshfitting/DATA/test_mmwhs/original_slice.nii.gz', dest='ref_path',
                        help='the file path of HR ref image')
    parser.add_argument('--output_path', type=str, default='results', dest='output_path',
                        help='the file save path of reconstructed result')
    parser.add_argument('-pre_trained_model', type=str, default=None, dest='pre_trained_model', help='t')
                 
    parser.add_argument('--scale', type=float, default='2', dest='scale',
                        help='the up-sampling scale k')
    parser.add_argument('-checkpoint_path', type=str, default='/home/langzhang/cest_imaging/ArSSR-main/checkpoint_model/', dest='checkpoint_path',
                        help='t')
    parser.add_argument('-epoch_s', type=int, default=400, dest='epoch_s',help='t')


    args = parser.parse_args()
    encoder_name = args.encoder_name
    decoder_name = args.decoder_name
    decoder_depth = args.decoder_depth
    decoder_width = args.decoder_width
    feature_dim = args.feature_dim
    PE_embedding = args.PE_embedding
    freq =  args.freq

    gpu = args.gpu
    is_gpu = args.is_gpu
    input_path = args.input_path
    ref_path = args.ref_path
    output_path = args.output_path
    scale = args.scale
    pre_trained_model = args.pre_trained_model
    experiment_name = args.experiment_name
    checkpoint_path = args.checkpoint_path
    epoch_s = args.epoch_s

    seed_everything(42)
    # -----------------------
    # model
    # -----------------------
    if is_gpu == 1 and torch.cuda.is_available():
        DEVICE = torch.device('cuda:{}'.format(str(gpu)))
    else:
        DEVICE = torch.device('cpu')
    ArSSR = model.ArSSR(encoder_name=encoder_name, decoder_name= decoder_name, feature_dim= feature_dim,
                        decoder_depth=int(decoder_depth / 2), decoder_width= decoder_width, freq= freq,
                        embedding = PE_embedding).to(DEVICE)
    # ArSSR.load_state_dict(torch.load(pre_trained_model, map_location=DEVICE))

    checkpoint_model = os.path.join(checkpoint_path, experiment_name, f'epoch-{epoch_s}.pth')   # TODO
    if os.path.exists(checkpoint_model):
        checkpoint = torch.load(checkpoint_model, map_location=DEVICE)
        ArSSR.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # -----------------------
    # SR
    # -----------------------
    filenames = os.listdir(input_path)
    print(filenames)
    for f in filenames: #tqdm(filenames)
        print(f)
        test_loader = data.loader_test(in_path_lr=r'{}/{}'.format(input_path, f), scale=scale)

        dir_path = r'{}/{}'.format(output_path, experiment_name)  # 替换为你想要创建的目录路径
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # read the dimension size and spacing of LR input image
        lr = nib.load(os.path.join(input_path, f))
        lr = lr.get_fdata()
        lr_size = lr.shape
        # lr_spacing = lr.GetSpacing()

        # then compute the dimension size and spacing of the HR image
        # hr_size = (np.array(lr_size) * scale).astype(int)
        # hr_spacing = np.array(lr_spacing) / scale

        #只沿着第单个方向放大
        hr_size = np.array(lr_size)
        hr_size[-1] *= scale
        hr_size = hr_size.astype(int)

        hr_size = (np.array(lr_size) * scale).astype(int)
        # hr_spacing = np.array(lr_spacing)
        # hr_spacing[-1] /= scale

        ArSSR.eval()
        with torch.no_grad():
            img_pre = np.zeros((hr_size[0] * hr_size[1] * hr_size[2], 1))  # 要预测的坐标点
            for i, (img_lr, xyz_hr) in enumerate(test_loader):
                img_lr = img_lr.unsqueeze(1).float().to(DEVICE)  # N×1×h×w×d
                for j in tqdm(range(hr_size[0])):
                    xyz_hr_patch = xyz_hr[:,
                                   j * hr_size[1] * hr_size[2]: j * hr_size[1] * hr_size[2] + hr_size[
                                       1] * hr_size[2], :].to(DEVICE)
                    img_pre_path = ArSSR(img_lr, xyz_hr_patch)
                    img_pre[
                    j * hr_size[1] * hr_size[2]: j * hr_size[1] * hr_size[2] + hr_size[1] * hr_size[
                        2]] = img_pre_path.cpu().detach().numpy().reshape(hr_size[1] * hr_size[2], 1)

            # 重塑 img_pre 以匹配高分辨率大小，并转换为 NumPy 数组
            img_pre = img_pre.reshape((hr_size[0], hr_size[1], hr_size[2]))

        print(img_pre.shape)

        ref_hr_image = nib.load(ref_path)
        # ref_hr_image = nib.load(os.path.join(ref_path, f))
        ref_hr_array = ref_hr_image.get_fdata()[:,:,:]  
        ref_hr_array_norm = normal(ref_hr_array)
        img_pre_norm = normal(img_pre)

        psnr_value = psnr(ref_hr_array_norm, img_pre_norm, data_range=1)
        ssim_value = ssim(ref_hr_array_norm, img_pre_norm, data_range=1, multichannel=True)
        mse_value = utils.compute_nmse(ref_hr_array_norm, img_pre_norm)

        print(f"PSNR: {psnr_value}")
        print(f"SSIM: {ssim_value}")
        print(f"NMSE: {mse_value}")

        # save file
        utils.write_img(vol=img_pre,
                        # ref_path=os.path.join(ref_path, f),
                        ref_path=ref_path,
                        out_path=os.path.join(dir_path, f'epoch_{epoch_s}_pred_{f}.nii'),
                        new_spacing=None)  # hr_spacing
        utils.write_img(vol=ref_hr_array,
                        # ref_path=os.path.join(ref_path, f),
                        ref_path=ref_path,
                        out_path=os.path.join(dir_path, f'epoch_{epoch_s}_ref_{f}.nii'),
                        new_spacing=None)  # hr_spacing
