import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import scipy.ndimage
import scipy.ndimage as ndi
def load_nii(filepath):
    nii = nib.load(filepath)
    data = nii.get_fdata()
    affine = nii.affine
    return data[:,:,1:], affine

def save_nii(data, affine, filepath):
    nii_img = nib.Nifti1Image(data, affine)
    nib.save(nii_img, filepath)

def process_data(input_folder, output_folders, target_sizes):
    filenames = os.listdir(input_folder)
    for size, output_folder in zip(target_sizes, output_folders):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for filename in tqdm(filenames, desc=f'Resizing to {size} frequencies'):
            if filename == 'volume_9.nii':
                filepath = os.path.join(input_folder, filename)
                data, affine = load_nii(filepath)  # (96, 96, 40)
                new_data = scipy.ndimage.zoom(data, (1, 1, size/data.shape[2]), order=3)  # 下采样 (96,96,40) 到 (96,96,size)
                new_filename = os.path.join(output_folder, filename)
                save_nii(new_data, affine, new_filename)

def process_4d_data(file_path, output_dir):
    # Load 4D NIfTI data
    img = nib.load(file_path)
    img_data = img.get_fdata()

    # Check the shape of the loaded data
    print(f"Original data shape: {img_data.shape}")

    # Extract (h, w, 1, 1) 3D slice
    h, w, d = img_data.shape
    if d < 1 :
        raise ValueError("The data does not have enough depth or time dimension for extraction.")

    slice_3d = img_data[:, :, :]
    print(f"Extracted 3D slice shape: {slice_3d.shape}")

    # Perform down-sampling (2x and 4x)
    downsampled_2x = ndi.zoom(slice_3d, 1 / 2, order=3)
    downsampled_4x = ndi.zoom(slice_3d, 1 / 4, order=3)

    print(f"2x downsampled shape: {downsampled_2x.shape}")
    print(f"4x downsampled shape: {downsampled_4x.shape}")

    # Save the original and downsampled slices as NIfTI files
    save_nii(slice_3d, img.affine, os.path.join(output_dir, 'original_slice.nii.gz'))
    save_nii(downsampled_2x, img.affine, os.path.join(output_dir, 'downsampled_2x.nii.gz'))
    save_nii(downsampled_4x, img.affine, os.path.join(output_dir, 'downsampled_4x.nii.gz'))


if __name__ == '__main__':
    # input_folder = '/data3/langzhang/cest_generative/cest_data/2_37/'  # 原始数据存放路径
    # output_folders = ['/data3/langzhang/cest_generative/cest_data/2_18/', '/data3/langzhang/cest_generative/cest_data/2_9/']  # 输出文件夹
    # target_sizes = [18, 9]  # 目标频偏点数
    # process_data(input_folder, output_folders, target_sizes)

    file_path = "/data3/langzhang/meshfitting/DATA/MM_WHS_2017_Dataset/mr_test/mr_test_2036_image.nii.gz"
    output_dir = "/data3/langzhang/meshfitting/DATA/test_mmwhs/"
    process_4d_data(file_path, output_dir)