import nibabel as nib
from scipy.ndimage import zoom
import os

def get_target_shape(input_path):
    img = nib.load(input_path)
    data = img.get_fdata()
    print(data.shape)
    return data.shape

def downsample_nifti(input_path, output_path, target_shape):
    # 加载NIfTI图像
    img = nib.load(input_path)
    data = img.get_fdata()
    
    # 计算每个维度的缩放因子
    zoom_factors = [n / o for n, o in zip(target_shape, data.shape)]
    
    # 使用zoom函数进行下采样，order=1代表双线性插值
    resampled_data = zoom(data, zoom_factors, order=1)
    
    # 创建一个新的NIfTI图像，使用原始图像的方向矩阵
    new_img = nib.Nifti1Image(resampled_data, img.affine, img.header)
    
    # 保存修改后的图像
    nib.save(new_img, output_path)

# 输入和输出文件路径
'''
这个输入数据应该是使用hush插值后的 —— final_process.py
然后转坐标 —— change_axis.py
再下采样xy轴得到输入数据 —— downsmaple_x2.py
'''
input_path = r"/data0/yurunyang/ResCNN_MLP_PE_normal/epoch_400_pred_gd_xinjianfeihou_frame0_0000_changeaxis.nii.gz"
output_path = r"/data0/yurunyang/ResCNN_MLP_PE_normal/gd_xinjianfeihou_frame0.nii.gz"

# 目标尺寸
scale = 2
target_shape = get_target_shape(input_path)
x = int(target_shape[0] / scale)
y = int(target_shape[1] / scale)
z = target_shape[2] 

target_shape = (x, y, z)
print("看看target：", target_shape)


# 调用函数
downsample_nifti(input_path, output_path, target_shape)


# 删除原文件
if os.path.exists(input_path):
    os.remove(input_path)
    print("文件已删除")
else:
    print("文件不存在")