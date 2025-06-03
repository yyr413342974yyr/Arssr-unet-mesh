import os
import numpy as np
import pydicom
import nibabel as nib

def process_dicoms_to_nifti(folder_path, output_folder):
    # 从文件夹读取所有 DICOM 文件
    dicom_files = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.endswith('.dcm')]

    # 总共有 740 个文件，需要处理成 20 个 nii 文件
    num_files = len(dicom_files)
    num_volumes = 20  # 总共需要创建 20 个 NIfTI 文件
    num_slices = 37  # 每个 NIfTI 文件包含 37 个切片

    # 以 20 个文件为周期，提取 37 个文件，总共进行 20 次
    for volume_index in range(num_volumes):
        # 提取特定间隔的 37 个文件
        slices = []
        for slice_index in range(num_slices):
            file_index = volume_index + slice_index * 20
            if file_index < num_files:  # 确保索引在范围内
                filepath = dicom_files[file_index]
                ds = pydicom.dcmread(filepath)
                slices.append(ds.pixel_array)
            else:
                break  # 如果索引超出范围，停止提取

        # 转换为 numpy 数组
        if slices:  # 确保列表不为空
            volume_array = np.stack(slices, axis=-1)  # 最后一个维度为 37

            # 创建 NIfTI 图像
            nifti_img = nib.Nifti1Image(volume_array, affine=np.eye(4))  # 使用单位仿射矩阵

            # 保存 NIfTI 文件
            output_path = os.path.join(output_folder, f'volume_{volume_index + 1}.nii')
            nib.save(nifti_img, output_path)
            print(f"Saved NIfTI volume {volume_index + 1} to {output_path}")

# 用法
process_dicoms_to_nifti('/data3/langzhang/cest_generative/cest_data/1/', '/data3/langzhang/cest_generative/cest_data/2/')
