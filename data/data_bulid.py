import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

def load_nii(file_path):
    """加载NIfTI文件并返回数据和仿射矩阵"""
    nii = nib.load(file_path)
    data = nii.get_fdata()
    affine = nii.affine
    return data[:,:,:], affine

def save_nii(data, affine, file_path):
    """保存NIfTI文件"""
    nii_img = nib.Nifti1Image(data, affine)
    nib.save(nii_img, file_path)

def chop_nii_dataset(file_names, input_path, output_path, num_patches=6, patch_size=(40, 40)):
    """随机裁剪NIfTI数据集，处理特定的文件列表"""
    for idx, f in tqdm(enumerate(file_names), total=len(file_names)):
        name = f.split('.')[0]

        img_in, affine = load_nii(os.path.join(input_path, f))
        h, w, d = img_in.shape

        for i in range(num_patches):
            x0 = np.random.randint(0, h - patch_size[0])
            y0 = np.random.randint(0, w - patch_size[1])
            patch_img = img_in[x0:x0 + patch_size[0], y0:y0 + patch_size[1], :]  # (40, 40, d)

            patch_save_path = os.path.join(output_path, f'{name}_{i}.nii.gz')
            save_nii(patch_img, affine, patch_save_path)


if __name__ == '__main__':
    input_path = '/data3/langzhang/cest_generative/cest_data/2_37/'
    train_output_path = '/data3/langzhang/cest_generative/cest_data/train_36_patches/'
    val_output_path = '/data3/langzhang/cest_generative/cest_data/val_36_patches/'
    test_output_path = '/data3/langzhang/cest_generative/cest_data/test_36_patches/'

    os.makedirs(train_output_path, exist_ok=True)
    os.makedirs(val_output_path, exist_ok=True)
    os.makedirs(test_output_path, exist_ok=True)

    # 数据划分逻辑
    filenames = sorted(os.listdir(input_path))
    train_files = filenames[:15]
    val_files = filenames[15:19]
    test_files = filenames[19:20]

    # 分别处理训练集、验证集和测试集
    chop_nii_dataset(train_files, input_path, train_output_path, num_patches=6, patch_size=(40, 40))
    chop_nii_dataset(val_files, input_path, val_output_path, num_patches=6, patch_size=(40, 40))
    chop_nii_dataset(test_files, input_path, test_output_path, num_patches=6, patch_size=(40, 40))