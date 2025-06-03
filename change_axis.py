import nibabel as nib
'''
这个可以进行轴翻转，同时进行nii.gz的格式转换
'''
def convert_and_modify_affine(file_path, output_path):
    # 加载NIfTI图像
    img = nib.load(file_path)

    # 显示原始方向矩阵
    original_affine = img.affine
    print("Original Affine Matrix:")
    print(original_affine)
    
    # 创建新的方向矩阵
    new_affine = original_affine.copy()
    new_affine[:, :3] = 0  # 将旋转缩放矩阵部分清零
    new_affine[0, 0] = -1  # 设置新的X轴方向
    new_affine[1, 1] = -1  # 设置新的Y轴方向
    new_affine[2, 2] = 1   # 设置新的Z轴方向
    new_affine[:, 3] = 0   # 将平移向量部分设置为零
    
    print("New Affine Matrix:")
    print(new_affine)
    
    # 创建一个新的NIfTI图像，使用修改后的方向矩阵
    new_img = nib.Nifti1Image(img.get_fdata(), new_affine, header=img.header)

    # 保存修改后的图像到.nii.gz格式
    nib.save(new_img, output_path)
    print("结束！")

# 指定原始文件路径和输出文件路径
file_path = r"/data0/yurunyang/ResCNN_MLP_PE_normal/epoch_400_pred_gd_xinjianfeihou_frame0_0000.nii.gz"
output_path = r"/data0/yurunyang/ResCNN_MLP_PE_normal/epoch_400_pred_gd_xinjianfeihou_frame0_0000_changeaxis.nii.gz"

convert_and_modify_affine(file_path, output_path)
