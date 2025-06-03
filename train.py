# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: train.py
#   Author  : Qing Wu
#   Email   : wuqing@shanghaitech.edu.cn
#   Date    : 2021/9/19
# -----------------------------------------
import os
import datetime
from data import data
import torch
from model import model
import argparse
import time
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import nibabel as nib
from utils import normal
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import utils
from tqdm import tqdm
import random
from model.Hloss_FF import HDRLoss_FF
import sys

def seed_everything(seed=0):
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

    parser.add_argument('-experiment_name', type=str, default='ResCNN_MLP_PE_Hash', dest='experiment_name', help='.')
                       
    # about ArSSR model
    parser.add_argument('-encoder_name', type=str, default='ResCNN', dest='encoder_name',
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
    parser.add_argument('-freq', type=int, default=9, dest='freq', help='')
                       

    # about training and validation data
    parser.add_argument('-hr_data_train', type=str, default='/data3/langzhang/meshfitting/DATA/MM_WHS_2017_Dataset/', dest='hr_data_train',
                        help='the file path of HR patches for training')
    parser.add_argument('-hr_data_val', type=str, default='/data3/langzhang/meshfitting/DATA/MM_WHS_2017_Dataset/', dest='hr_data_val',
                        help='the file path of HR patches for validation')

    # about training hyper-parameters
    parser.add_argument('-lr', type=float, default=1e-4, dest='lr',
                        help='the initial learning rate')
    parser.add_argument('-lr_decay_epoch', type=int, default=200, dest='lr_decay_epoch',
                        help='learning rate multiply by 0.5 per lr_decay_epoch .')
    parser.add_argument('-epoch', type=int, default=2600, dest='epoch',
                        help='the total number of epochs for training')
    parser.add_argument('-summary_epoch', type=int, default=200, dest='summary_epoch',    
                        help='the current model will be saved per summary_epoch')
    parser.add_argument('-print_fre', type=int, default=5, dest='print_fre',
                        help='the current model will be saved per print_fre')
    parser.add_argument('-bs', type=int, default=1, dest='batch_size',
                        help='the number of LR-HR patch pairs (i.e., N in Equ. 3)')
    parser.add_argument('-ss', type=int, default=8000, dest='sample_size',
                        help='the number of sampled voxel coordinates (i.e., K in Equ. 3)')
    parser.add_argument('-gpu', type=int, default=3, dest='gpu',
                        help='the number of GPU')
    parser.add_argument('-checkpoint_path', type=str, default='/home/langzhang/cest_imaging/ArSSR-org/checkpoint_model/', dest='checkpoint_path',
                        help='t')
    parser.add_argument('-epoch_s', type=int, default=0, dest='epoch_s',help='t')
                        

    args = parser.parse_args()
    encoder_name = args.encoder_name
    decoder_name = args.decoder_name
    decoder_depth = args.decoder_depth
    decoder_width = args.decoder_width
    feature_dim = args.feature_dim
    PE_embedding = args.PE_embedding
    freq =  args.freq

    experiment_name = args.experiment_name
    hr_data_train = args.hr_data_train
    hr_data_val = args.hr_data_val

    lr = args.lr
    lr_decay_epoch = args.lr_decay_epoch
    epoch = args.epoch
    summary_epoch = args.summary_epoch
    batch_size = args.batch_size
    sample_size = args.sample_size
    print_fre = args.print_fre
    gpu = args.gpu
    checkpoint_path = args.checkpoint_path
    epoch_s = args.epoch_s

    writer = SummaryWriter(f'./log/{experiment_name}')

    seed_everything(42)
    # -----------------------
    # display parameters
    # -----------------------
    run_id = str(os.getpid())
    print('Experiment Name:', experiment_name)
    print("id=", run_id)
    save_model_path = os.path.join(os.getcwd(),'checkpoint_model', experiment_name)
    print('save_model_path=', save_model_path)
    save_fig_path = os.path.join(os.getcwd(),'fig', experiment_name)
    print('save_fig_path=', save_fig_path)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)

    print('Parameter Settings')
    print('')
    print('------------File------------')
    print('hr_data_train: {}'.format(hr_data_train))
    print('hr_data_val: {}'.format(hr_data_val))
    print('------------Train-----------')
    print('lr: {}'.format(lr))
    print('batch_size_train: {}'.format(batch_size))
    print('sample_size: {}'.format(sample_size))
    print('gpu: {}'.format(gpu))
    print('epochs: {}'.format(epoch))
    print('summary_epoch: {}'.format(summary_epoch))
    print('lr_decay_epoch: {}'.format(lr_decay_epoch))
    print('------------Model-----------')
    print('encoder_name : {}'.format(encoder_name))
    print('decoder_name: {}'.format(decoder_name))
    print('decoder feature_dim: {}'.format(feature_dim))
    print('decoder depth: {}'.format(decoder_depth))
    print('decoder width: {}'.format(decoder_width))

    # -----------------------
    # load data
    # -----------------------
    train_loader = data.loader_data(in_path_hr=hr_data_train, batch_size=batch_size,
                                     sample_size=sample_size, mode="train")
    val_loader = data.loader_data(in_path_hr=hr_data_val, batch_size=1,
                                   sample_size=sample_size, mode="val")
    # -----------------------
    # model & optimizer
    # -----------------------
    DEVICE = torch.device('cuda:{}'.format(str(gpu) if torch.cuda.is_available() else 'cpu'))
    ArSSR = model.ArSSR_Hash(encoder_name=encoder_name, decoder_name= decoder_name, feature_dim= feature_dim,
                        decoder_depth=int(decoder_depth / 2), decoder_width= decoder_width, freq= freq,
                        embedding = PE_embedding).to(DEVICE)

    loss_fun = torch.nn.L1Loss()
    loss_fun_2 = HDRLoss_FF()
    optimizer = torch.optim.Adam(params=ArSSR.parameters(), lr=lr)

    # Load model and optimizer state if a checkpoint exists
    checkpoint_model = os.path.join(checkpoint_path, experiment_name, f'epoch-{epoch_s}.pth')   # TODO
    if os.path.exists(checkpoint_model):
        checkpoint = torch.load(checkpoint_model, map_location=DEVICE)
        ArSSR.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        start_epoch = 0
        print(f"Loaded checkpoint from epoch {start_epoch}")

    ## 预训练模型
    # checkpoint_model = "/home/langzhang/cest_imaging/ArSSR-main/pre_trained_models/ArSSR_ResCNN.pkl"
    # ArSSR.load_state_dict(torch.load(checkpoint_model, map_location=DEVICE))

    # -----------------------
    # training & validation
    # -----------------------
    for e in range(start_epoch, epoch):
        ArSSR.train()
        loss_train =  0
        for i, (img_lr, xyz_hr, img_hr, xyz_lr) in enumerate(train_loader):
            optimizer.zero_grad()
            # forward
            img_lr = img_lr.unsqueeze(1).to(DEVICE).float()  # N×1×h×w×d
            img_hr = img_hr.to(DEVICE).float().view(batch_size, -1).unsqueeze(-1)  # N×K×1 (K Equ. 3)
            H, W, D = 20, 20, 20
            xyz_hr = xyz_hr.view(batch_size, -1, 3).to(DEVICE).float()  # N×K×3
            img_pre = ArSSR(img_lr, xyz_hr, H, W, D)  # N×K×1
            loss = loss_fun(img_pre, img_hr)

            # frequency
            # freq_hr = torch.fft.fftn(img_hr)
            # freq_pred = torch.fft.fftn(img_pre)
            # # loss2 = loss_fun(freq_pred, freq_hr)
            # loss2, reg = loss_fun_2(freq_pred, freq_hr, xyz_hr)

            # loss = loss1 + loss2
            # backward
            loss.backward()
            optimizer.step()
            # record and print loss
            loss_train += loss.item()
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            if e == 0:
                print('(TRAIN) Epoch[{}/{}], Steps[{}/{}], Lr:{}, Loss:{:.5f}'.format(e + 1,
                                                                                    epoch,
                                                                                    i+1 ,
                                                                                    len(train_loader),
                                                                                    current_lr,
                                                                                    loss.item()))
            else:
                sys.stdout.write('\r(TRAIN) Epoch[{}/{}], Steps[{}/{}], Lr:{}, Loss:{:.5f}'.format(
                e + 1, epoch, i + 1, len(train_loader), current_lr, loss.item()))
                sys.stdout.flush()                                                                
        if e > 0:
            sys.stdout.write('\n')

        writer.add_scalar('MES_train', loss_train / len(train_loader), e + 1)

        ArSSR.eval()
        with torch.no_grad():
            loss_val = 0
            for i, (img_lr, xyz_hr, img_hr, xyz_lr) in enumerate(val_loader):
                img_lr = img_lr.unsqueeze(1).to(DEVICE).float()  # N×1×h×w×d
                xyz_hr = xyz_hr.view(1, -1, 3).to(DEVICE).float()  # N×Q×3 (Q=H×W×D)
                H, W, D = 20,20,20
                img_hr = img_hr.to(DEVICE).float().view(1, -1).unsqueeze(-1)  # N×Q×1 (Q=H×W×D)
                img_pre = ArSSR(img_lr, xyz_hr, H, W, D)  # N×Q×1 (Q=H×W×D)

                loss = loss_fun(img_pre, img_hr)       

                # # frequency
                # freq_hr = torch.fft.fftn(img_hr)
                # freq_pred = torch.fft.fftn(img_pre)
                # # loss2 = loss_fun(freq_pred, freq_hr)
                # loss2, reg = loss_fun_2(freq_pred, freq_hr, xyz_hr)

                loss_val += loss 
            
                # save validation
                if (e + 1) % summary_epoch == 0:
                    # save model
                    sv_file = {
                        'model_state_dict': ArSSR.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': e + 1
                    }
                    torch.save(sv_file,
                               os.path.join(checkpoint_path, experiment_name, 'epoch-{}.pth'.format(e + 1)))

        writer.add_scalar('MES_val', loss_val / len(val_loader), e + 1)

        # 测试单个数据
        if e % summary_epoch == 0:
            test_loader = data.loader_test("/data3/langzhang/meshfitting/DATA/test_mmwhs/downsampled_2x.nii.gz", scale=2)
            lr_size = [128, 128, 70]
            scale = 2
            hr_size = np.array(lr_size)
            hr_size *= scale

            with torch.no_grad():
                img_pre = np.zeros((hr_size[0] * hr_size[1] * hr_size[2], 1))  # 要预测的坐标点
                for i, (img_lr, xyz_hr) in enumerate(test_loader):
                    img_lr = img_lr.unsqueeze(1).float().to(DEVICE)  # N×1×h×w×d
                    for j in tqdm(range(hr_size[0])):
                        xyz_hr_patch = xyz_hr[:,
                                       j * hr_size[1] * hr_size[2]: j * hr_size[1] * hr_size[2] + hr_size[
                                           1] * hr_size[2], :].to(DEVICE)
                        H, W, D = 20,20,20
                        img_pre_path = ArSSR(img_lr, xyz_hr_patch, H, W, D)
                        img_pre[
                        j * hr_size[1] * hr_size[2]: j * hr_size[1] * hr_size[2] + hr_size[1] * hr_size[
                            2]] = img_pre_path.cpu().detach().numpy().reshape(hr_size[1] * hr_size[2], 1)

                # 重塑 img_pre 以匹配高分辨率大小，并转换为 NumPy 数组
                img_pre = img_pre.reshape((hr_size[0], hr_size[1], hr_size[2]))

            ## eval indice
            ref_path = "/data3/langzhang/meshfitting/DATA/test_mmwhs/original_slice.nii.gz"
            ref_hr_image = nib.load(ref_path)
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
                            ref_path=ref_path,
                            out_path=os.path.join(save_fig_path, f'epoch_{e}_pred.nii'),
                            new_spacing=None)  # hr_spacing
            utils.write_img(vol=ref_hr_array,
                            ref_path=ref_path,
                            out_path=os.path.join(save_fig_path, f'epoch_{e}_ref.nii'),
                            new_spacing=None)  # hr_spacing

        # learning rate decays by half every some epochs.
        if (e + 1) % lr_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

    writer.flush()
