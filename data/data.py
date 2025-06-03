# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: dataset.py
#   Author  : Qing Wu
#   Email   : wuqing@shanghaitech.edu.cn
#   Date    : 2021/9/19
# -----------------------------------------
import os
import nibabel as nib
from tqdm import tqdm
from matplotlib import pyplot as plt
import random
import SimpleITK as sitk
from torch.utils import data
import scipy.ndimage as ndi  # 使用 scipy 的 ndimage 替代
import numpy as np
import utils
# -----------------------
# Training data
# -----------------------

class ImgTrain(data.Dataset):
    def __init__(self, in_path_hr, sample_size, is_train):
        self.is_train = is_train
        self.sample_size = sample_size
        self.patch_hr = utils.read_img(in_path=in_path_hr)

    def __len__(self):
        return len(self.patch_hr)

    def __getitem__(self, item):
        patch_hr = self.patch_hr[item]
        # norm
        patch_hr = utils.normal(patch_hr)
        # randomly get an up-sampling scale from [2, 4]
        s = np.round(random.uniform(2, 4 + 0.04), 1)
        # compute the size of HR patch according to the scale
        hr_d, hr_h, hr_w = np.minimum((np.array([10, 10, 10]) * s).astype(int), patch_hr.shape)
        # generate HR patch by cropping
        patch_hr = patch_hr[:hr_d, :hr_h, :hr_w]      # (h,w,d)
        # simulated LR patch by down-sampling HR patch        # (20~40)之间任意数 ,  s=(2~4)之间任意数
        patch_lr = ndi.interpolation.zoom(patch_hr, 1 / s, order=3)         # (10,10,10)
        # generate coordinate set
        xyz_hr = utils.make_coord(patch_hr.shape, flatten=True)

        # randomly sample voxel coordinates
        # if self.is_train:
        #     sample_indices = np.random.choice(len(xyz_hr), self.sample_size, replace=False)
        #     xyz_hr = xyz_hr[sample_indices]
        #     patch_hr_sample = patch_hr.reshape(-1, 1)[sample_indices]   # (K,1)

        return patch_lr, xyz_hr, patch_hr

def loader_train(in_path_hr, batch_size, sample_size, is_train):
    """
    :param in_path_hr: the path of HR patches
    :param batch_size: N in Equ. 3
    :param sample_size: K in Equ. 3
    :param is_train:
    :return:
    """
    return data.DataLoader(
        dataset=ImgTrain(in_path_hr=in_path_hr, sample_size=sample_size, is_train=is_train),
        batch_size=batch_size,
        shuffle=is_train
    )

class cardio_acdc(data.Dataset):
    def __init__(self, data_dir, sample_size, mode):
        self.mode = mode
        self.sample_size = sample_size
        # Load and split 4D data along the frame dimension
        self.patch_hr_list = self.load_and_split_data(data_dir)

    def __len__(self):
        return len(self.patch_hr_list)

    def __getitem__(self, item):
        patch_hr = self.patch_hr_list[item]
        # norm
        patch_hr = utils.normal(patch_hr)
        # randomly get an up-sampling scale from [2, 4]
        s = np.round(random.uniform(2, 4 + 0.04), 1)
        # compute the size of HR patch according to the scale
        hr_d, hr_h, hr_w = np.minimum((np.array([10, 10, 5]) * s).astype(int), patch_hr.shape)
        # generate HR patch by cropping
        patch_hr = patch_hr[:hr_d, :hr_h, :hr_w]
        # simulated LR patch by down-sampling HR patch        # (20~40)之间任意数 ,  s=(2~4)之间任意数
        patch_lr = ndi.interpolation.zoom(patch_hr, 1 / s, order=3)         # (10,10,10)
        # generate coordinate set
        xyz_hr = utils.make_coord(patch_hr.shape, flatten=True)

        # randomly sample voxel coordinates
        # if self.mode == 'train':
        #     sample_indices = np.random.choice(len(xyz_hr), self.sample_size, replace=False)
        #     xyz_hr = xyz_hr[sample_indices]
        #     patch_hr = patch_hr.reshape(-1, 1)[sample_indices]

        return patch_lr, xyz_hr, patch_hr

    def load_and_split_data(self, data_dir):
        patch_hr_list = []

        # Assuming each folder corresponds to one sample
        folder_names = sorted(os.listdir(data_dir))

        if self.mode == 'train':
            selected_folders = folder_names[:85]
        elif self.mode == 'val':
            selected_folders = folder_names[85:95]
        else:  # mode == 'test'
            selected_folders = folder_names[95:]
        for folder in tqdm(selected_folders, desc='Loading data'):
            folder_path = os.path.join(data_dir, folder)
            file_names = [f for f in os.listdir(folder_path) if f.endswith('_4d.nii.gz')]
            for f in file_names:
                img_path = os.path.join(folder_path, f)
                img_data = self.load_nii(img_path)  # Load 4D NIfTI data
                # print(img_data.shape)
                for frame_idx in range(img_data.shape[3]):
                    frame_data = img_data[:, :, :, frame_idx]  # 提取每帧数据
                    frame_data_padded = self.pad_to_target_slices(frame_data, 20)
                    patches = self.chop_nii_dataset(frame_data_padded)  # Chop into patches
                    patch_hr_list.extend(patches)

        return patch_hr_list

    def pad_to_target_slices(self, array, target_slices):
        current_slices = array.shape[2]
        if current_slices >= target_slices:
            return array

        # 复制填充
        # num_repeats = target_slices // current_slices
        # remainder = target_slices % current_slices
        # padded_array = np.concatenate([array] * num_repeats + [array[:, :, :remainder]], axis=2)

        # 零填充
        padding_slices = target_slices - current_slices
        padded_array = np.pad(array, [(0, 0), (0, 0), (0, padding_slices)], mode='constant')

        return padded_array

    def load_nii(self, file_path):
        img = nib.load(file_path)
        img_data = img.get_fdata()
        return img_data

    def chop_nii_dataset(self, img_data, num_patches=36, patch_size=(40, 40)):
        patches = []

        h, w, d = img_data.shape  # Assuming the 3D structure is (h, w, d)

        for i in range(num_patches):
            x0 = np.random.randint(0, h - patch_size[0])
            y0 = np.random.randint(0, w - patch_size[1])
            patch_img = img_data[x0:x0 + patch_size[0], y0:y0 + patch_size[1], :]  # (40, 40, d)
            patches.append(patch_img)

        return patches

class MM_WHS_acdc(data.Dataset):
    def __init__(self, data_dir, sample_size, mode):
        self.mode = mode
        self.sample_size = sample_size
        # Load and split 4D data along the frame dimension
        self.patch_hr_list = self.load_and_split_data(data_dir)

    def __len__(self):
        return len(self.patch_hr_list)

    def __getitem__(self, item):
        patch_hr = self.patch_hr_list[item]
        # norm
        # patch_hr = utils.normal(patch_hr)
        # randomly get an up-sampling scale from [2, 4]
        s = np.round(random.uniform(2, 4 + 0.04), 1)
        # compute the size of HR patch according to the scale
        hr_d, hr_h, hr_w = np.minimum((np.array([10, 10, 10]) * s).astype(int), patch_hr.shape)
        # generate HR patch by cropping
        patch_hr = patch_hr[:hr_d, :hr_h, :hr_w]        #(H,W,D)
        # simulated LR patch by down-sampling HR patch        # (20~40)之间任意数 ,  s=(2~4)之间任意数
        patch_lr = ndi.interpolation.zoom(patch_hr, 1 / s, order=3)         # (10,10,10)
        # generate coordinate set
        xyz_hr = utils.make_coord(patch_hr.shape, flatten=True)
        xyz_lr = utils.make_coord(patch_lr.shape, flatten=True)
        # randomly sample voxel coordinates
        if self.mode == 'train':
            sample_indices = np.random.choice(len(xyz_hr), self.sample_size, replace=False)
            xyz_hr = xyz_hr[sample_indices]
            patch_hr = patch_hr.reshape(-1, 1)[sample_indices]   # (K,1)
        
        return patch_lr, xyz_hr, patch_hr, xyz_lr

    def load_and_split_data(self, data_dir):
        patch_hr_list = []

        # Assuming each folder corresponds to one sample
        folder_names1 = sorted(os.path.join(data_dir, 'mr_train', f) for f in os.listdir(os.path.join(data_dir, 'mr_train')) if f.endswith('image.nii.gz'))
        folder_names2 = sorted(os.path.join(data_dir, 'mr_test', f) for f in os.listdir(os.path.join(data_dir, 'mr_test')))
        folder_names =  folder_names1 + folder_names2

        if self.mode == 'train':
            selected_folders = folder_names[:50]
        elif self.mode == 'val':
            selected_folders = folder_names[50:55]
        else:  # mode == 'test'
            selected_folders = folder_names[55:]

        for folder in tqdm(selected_folders, desc='Loading data'):
            img_data = self.load_nii(folder)  # Load 4D NIfTI data
            # print(img_data.shape)
            patches = self.chop_nii_dataset(img_data)  # Chop into patches
            patch_hr_list.extend(patches)

        return patch_hr_list

    def load_nii(self, file_path):
        img = nib.load(file_path)
        img_data = img.get_fdata()
        return img_data

    def chop_nii_dataset(self, img_data, num_patches=18, patch_size=(40, 40, 40)):
        patches = []

        h, w, d = img_data.shape  # Assuming the 3D structure is (h, w, d)

        for i in range(num_patches):
            x0 = np.random.randint(0, h - patch_size[0])
            y0 = np.random.randint(0, w - patch_size[1])
            z0 = np.random.randint(0, d - patch_size[2])
            patch_img = img_data[x0:x0 + patch_size[0], y0:y0 + patch_size[1], z0:z0 + patch_size[2]]  # (40, 40, 40)
            patches.append(patch_img)

        return patches

class ACDC_sinle(data.Dataset):
    def __init__(self, data_path):
        # Load and split 4D data along the frame dimension
        self.hr_list = self.load_and_split_data(data_path)
        



    def __len__(self):
        return len(self.patch_hr_list)

    def __getitem__(self, item):
        patch_hr = self.patch_hr_list[item]
        # norm
        patch_hr = utils.normal(patch_hr)
        # randomly get an up-sampling scale from [2, 4]
        s = np.round(random.uniform(2, 4 + 0.04), 1)
        # compute the size of HR patch according to the scale
        hr_d, hr_h, hr_w = np.minimum((np.array([10, 10, 5]) * s).astype(int), patch_hr.shape)
        # generate HR patch by cropping
        patch_hr = patch_hr[:hr_d, :hr_h, :hr_w]
        # simulated LR patch by down-sampling HR patch        # (20~40)之间任意数 ,  s=(2~4)之间任意数
        patch_lr = ndi.interpolation.zoom(patch_hr, 1 / s, order=3)         # (10,10,10)
        # generate coordinate set
        xyz_hr = utils.make_coord(patch_hr.shape, flatten=True)

        # randomly sample voxel coordinates
        # if self.mode == 'train':
        #     sample_indices = np.random.choice(len(xyz_hr), self.sample_size, replace=False)
        #     xyz_hr = xyz_hr[sample_indices]
        #     patch_hr = patch_hr.reshape(-1, 1)[sample_indices]

        return patch_lr, xyz_hr, patch_hr

    def load_and_split_data(self, data_dir):
        patch_hr_list = []

        # Assuming each folder corresponds to one sample
        folder_names = sorted(os.listdir(data_dir))

        if self.mode == 'train':
            selected_folders = folder_names[:85]
        elif self.mode == 'val':
            selected_folders = folder_names[85:95]
        else:  # mode == 'test'
            selected_folders = folder_names[95:]
        for folder in tqdm(selected_folders, desc='Loading data'):
            folder_path = os.path.join(data_dir, folder)
            file_names = [f for f in os.listdir(folder_path) if f.endswith('_4d.nii.gz')]
            for f in file_names:
                img_path = os.path.join(folder_path, f)
                img_data = self.load_nii(img_path)  # Load 4D NIfTI data
                # print(img_data.shape)
                for frame_idx in range(img_data.shape[3]):
                    frame_data = img_data[:, :, :, frame_idx]  # 提取每帧数据
                    frame_data_padded = self.pad_to_target_slices(frame_data, 20)
                    patches = self.chop_nii_dataset(frame_data_padded)  # Chop into patches
                    patch_hr_list.extend(patches)

        return patch_hr_list

    def pad_to_target_slices(self, array, target_slices):
        current_slices = array.shape[2]
        if current_slices >= target_slices:
            return array

        # 复制填充
        # num_repeats = target_slices // current_slices
        # remainder = target_slices % current_slices
        # padded_array = np.concatenate([array] * num_repeats + [array[:, :, :remainder]], axis=2)

        # 零填充
        padding_slices = target_slices - current_slices
        padded_array = np.pad(array, [(0, 0), (0, 0), (0, padding_slices)], mode='constant')

        return padded_array

    def load_nii(self, file_path):
        img = nib.load(file_path)
        img_data = img.get_fdata()
        return img_data

    def chop_nii_dataset(self, img_data, num_patches=36, patch_size=(40, 40)):
        patches = []

        h, w, d = img_data.shape  # Assuming the 3D structure is (h, w, d)

        for i in range(num_patches):
            x0 = np.random.randint(0, h - patch_size[0])
            y0 = np.random.randint(0, w - patch_size[1])
            patch_img = img_data[x0:x0 + patch_size[0], y0:y0 + patch_size[1], :]  # (40, 40, d)
            patches.append(patch_img)

        return patches



def loader_data(in_path_hr, batch_size, sample_size, mode='test'):

    """
    :param in_path_hr: the path of HR patches
    :param batch_size: N in Equ. 3
    :param sample_size: K in Equ. 3
    :param is_train:
    :return:
    """
    if mode == 'test':
        flash = False
    else:
        flash = True
    return data.DataLoader(
        dataset=MM_WHS_acdc(data_dir=in_path_hr, sample_size=sample_size, mode=mode),
        batch_size=batch_size,
        shuffle=flash
    )


# -----------------------
# Testing data
# -----------------------

class ImgTest(data.Dataset):
    def __init__(self, in_path_lr, scale):
        self.img_lr = []
        self.xyz_hr = []

        # load lr image
        # lr_vol = sitk.GetArrayFromImage(sitk.ReadImage(in_path_lr))
        lr_vol = utils.read_img(in_path= in_path_lr)
        # lr_vol = utils.normal(lr_vol)

        self.img_lr.append(lr_vol)

        for img_lr in self.img_lr:
            temp_size = np.array(img_lr.shape).astype(float) #lr尺寸
            temp_size *= scale  # 再 ×scale
            temp_size = list(temp_size.astype(int))

            # 沿着一个方向SR
            # temp_size = np.array(img_lr.shape).astype(float) # lr尺寸
            # temp_size[2] *= scale
            # temp_size =list(temp_size.astype(int))  # HR坐标尺寸

            self.xyz_hr.append(utils.make_coord(temp_size, flatten=True))

    def __len__(self):
        return len(self.img_lr)

    def __getitem__(self, item):
        return self.img_lr[item], self.xyz_hr[item]

def loader_test(in_path_lr, scale):
    return data.DataLoader(
        dataset=ImgTest(in_path_lr=in_path_lr, scale=scale),
        batch_size=1,
        shuffle=False
    )


if __name__ == '__main__':
    hr_data_train = "/data3/langzhang/meshfitting/DATA/MM_WHS_2017_Dataset/"

    train_loader = loader_data(in_path_hr=hr_data_train, batch_size=1, sample_size=8000, mode="test")

    for i, (img_lr, xyz_hr, img_hr) in enumerate(train_loader):
        if i == 0:  # 只可视化第一批数据中的第一个样本
            img_hr = img_hr.numpy().squeeze()  # 假设 img_hr 是批中的高分辨率图像，需要从 torch.Tensor 转换
            img_lr = img_lr.numpy().squeeze()  # 假设 img_lr 是对应的低分辨率图像
            xyz_hr = xyz_hr.numpy().squeeze()  # 体素坐标

            # Normalize the images for visualization
            img_hr = utils.normal(img_hr)
            img_lr = utils.normal(img_lr)

            # Calculate the scale factor between LR and HR images
            scale_factor = img_lr.shape[1] / img_hr.shape[1]

            # Define the slice indices for visualization
            slice_indices_hr = [0, img_hr.shape[2] // 2, img_hr.shape[2] - 1]
            slice_indices_lr = [int(idx * scale_factor) for idx in slice_indices_hr]

            # Create the figure and axes for visualization
            fig, axs = plt.subplots(nrows=len(slice_indices_hr), ncols=2, figsize=(10, 15))

            for index, (slice_hr, slice_lr) in enumerate(zip(slice_indices_hr, slice_indices_lr)):
                # Visualize HR slice
                axs[index, 0].imshow(img_hr[:, :, slice_hr], cmap='gray')
                axs[index, 0].set_title(f'High-Resolution Patch - Slice {slice_hr}')
                axs[index, 0].axis('off')

                # Visualize LR slice
                axs[index, 1].imshow(img_lr[:, :, slice_lr], cmap='gray')
                axs[index, 1].set_title(f'Low-Resolution Patch - Slice {slice_lr}')
                axs[index, 1].axis('off')

            plt.tight_layout()
            plt.savefig('sample_plot.png')

            break  # 只展示一个样本就足够了


