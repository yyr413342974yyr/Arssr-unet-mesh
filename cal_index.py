import os
import pickle
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
# import lpips
import nibabel as nib
from scipy.ndimage import filters
from scipy.special import gammaln
from scipy.stats import genpareto

from PIL import Image, ImageChops

import cv2

from scipy.ndimage import binary_opening,binary_closing,binary_erosion
import matplotlib.pyplot as plt
from skimage.restoration import estimate_sigma,denoise_nl_means
from skimage import img_as_float

# from dipy.denoise.nlmeans import nlmeans
# from dipy.denoise.noise_estimate import estimate_sigma


def compute_nmse(true_img, pred_img):
    mse = np.sum((true_img - pred_img) ** 2)
    nmse_value = mse / np.sum(true_img ** 2)
    return nmse_value


def compress_image(input_path, output_path):
    # 打开图像
    original_image = Image.open(input_path)

    # 获取原始图像的宽度和高度
    width, height = original_image.size

    # 将图像压缩为原始尺寸的一半
    compressed_image = original_image.resize((width // 2, height // 2))

    # 保存压缩后的图像
    compressed_image.save(output_path)

    return compressed_image


def calculate_niqe(image):
    image = img_as_float(image)
    h, w = image.shape[:2]
    block_size = 96
    strides = 32
    features = []

    for i in range(0, h - block_size + 1, strides):
        for j in range(0, w - block_size + 1, strides):
            block = image[i:i + block_size, j:j + block_size]
            mu = np.mean(block)
            sigma = np.std(block)
            filtered_block = filters.gaussian_filter(block, sigma)
            shape, _, scale = genpareto.fit(filtered_block.ravel(), floc=0)
            feature = [mu, sigma, shape, scale, gammaln(1 / shape)]
            features.append(feature)

    features = np.array(features)
    model_mean = np.zeros(features.shape[1])
    model_cov_inv = np.eye(features.shape[1])
    quality_scores = []

    for feature in features:
        score = (feature - model_mean) @ model_cov_inv @ (feature - model_mean).T
        quality_scores.append(score)

    return np.mean(quality_scores)
def pad_nii(original_array,target_shape):
    # 假设你有一个形状为 (x, y, z) 的数组
    # original_array = np.random.rand(3, 120, 80)

    # 定义目标形状
    # target_shape = ( 192, 192)

    # 计算每个维度的填充量
    pad_width = []  # 不对第一个维度进行填充
    for dim_size, target_size in zip(original_array.shape, target_shape):
        before_pad = max(0, (target_size - dim_size) // 2)
        after_pad = max(0, target_size - dim_size - before_pad)
        pad_width.append((before_pad, after_pad))
    # print(pad_width)
    # 使用 np.pad() 进行填充
    padded_array = np.pad(original_array, pad_width , mode='constant', constant_values=0)
    # print("原始数组形状：", original_array.shape)
    # print("填充后数组形状：", padded_array.shape)
    return padded_array


def norm_img(img):
    max_value = torch.max(img)
    min_value = torch.min(img)
    norm_img = (img - min_value) / (max_value - min_value)
    return norm_img

def eval_slice(true_img, pred_img,mask):


    # true_image = Image.open(true_img_path)
    # pred_image = Image.open(pred_img_path)
    # true_img=np.array(true_image)
    # pred_img = np.array(pred_image)#.mean(axis=-1).astype(np.uint8)

    true_img = true_img * mask
    pred_img = pred_img * mask
    # # 归一化
    true_img = ((true_img - true_img.min()) / (
            true_img.max() - true_img.min()))  # * 255.astype(np.uint8)
    pred_img = ((pred_img - pred_img.min()) / (
            pred_img.max() - pred_img.min()))  # * 255.astype(np.uint8)




    if np.all(pred_img >= 0) and np.all(pred_img <= 1):
        print(f"All values in pred_img are between 0 and 1.")
    else:
        print(f"Some values in pred_img are outside the range [0, 1].max:{pred_img.max()},min:{pred_img.min()}")
        pred_img = np.clip(pred_img, 0, 1)

    if np.all(true_img >= 0) and np.all(true_img <= 1):
        print(f"All values in true_img are between 0 and 1.")
    else:
        print(f"Some values in true_img are outside the range [0, 1].max:{true_img.max()},min:{true_img.min()}")
        true_img = np.clip(true_img, 0, 1)

    psnr_value = psnr(true_img, pred_img)  # Assuming images are normalized to [0,1]
    ssim_value, _ = ssim(true_img, pred_img, full=True,multichannel=True)  ## Assuming images are normalized to [0,1]#multichannel=True三通道设置
    nmse_value = compute_nmse(true_img, pred_img)


    # ###########lpips、NIQE超分指标计算
    # #转换为tensor
    # true_tensor =torch.tensor(true_img) / 255.0
    # pred_tensor =torch.tensor(pred_img)/ 255.0
    # loss_fn = lpips.LPIPS(net='vgg')
    # lpips_value=loss_fn(true_tensor,pred_tensor).item()
    #
    # NIQE_value=calculate_niqe(pred_img)





    # print(f"lpips:{lpips_value}")
    # # print("lpips:{}".format(lpips_value.item()))
    # print(f"NIQE: {NIQE_value}")
    # print(f"NMSE: {nmse_value}")
    # np.save(os.path.join(this_sample_dir,f'fa_{slice_idx}.npy'), pred_img)
    return psnr_value,ssim_value,nmse_value#lpips_value,NIQE_value


def differ_image(true_img_path, pred_img_path):
    # 读取两张图片
    image1 = Image.open(true_img_path)
    image2 = Image.open(pred_img_path)



    filename=pred_img_path.split("/")[-1]
    prename=filename.split(".")[0]


    root_path=os.path.dirname(pred_img_path)
    print(root_path)

    # 将图像转为 NumPy 数组
    array1 = np.array(image1)
    array2 = np.array(image2)


    # 计算两张图片的差值
    # difference_array = np.abs(array1 - array2)
    difference_image = ImageChops.difference(image1,image2)

    # 将数据转为uint8类型（0-255范围内），因为PIL需要这个数据类型
    # difference_array = ((difference_array - difference_array.min()) / (difference_array.max() - difference_array.min()) * 255)

    # 将差值数组转为图像
    # difference_image = Image.fromarray(difference_array.astype(np.uint8))



    # 保存差值图像或者显示
    # difference_image.save('difference_image.jpg')
    # print(r"{}/{}_differ.png".format(root_path, prename))
    difference_image.save(r"{}/{}_differ.jpg".format(root_path, prename))
    # difference_image.show()


    return difference_image


def get_mask(path,slice):
    # mask 评估需要
    mask_data = nib.load(path)#'/data3/langzhang/dtidata_test/578057/mask.nii.gz'
    mask_data = mask_data.get_fdata(dtype=np.float32)
    mask_data = mask_data[:, :, slice]
    mask_data = pad_nii(mask_data,(192,192))  # padding
    max_value = mask_data.max()
    if max_value != 0:  # To avoid division by zero
        mask_data /= max_value

    return mask_data

def mask_seg(mask,img):
    img=np.array(img)
    img_R=(img[:,:,0]/ 255.0)*mask
    img_G = (img[:, :, 1]/ 255.0) * mask
    img_B = (img[:, :, 2]/ 255.0) * mask
    seg_img =np.stack((img_R, img_G,img_B), axis=-1)



    # 将数据转换为8位整数
    seg_img = (seg_img * 255.0).astype(np.uint8)


    # seg_img= ((seg_img - seg_img.min()) / (seg_img.max() - seg_img.min()) * 255).astype(np.uint8)
    seg_img = Image.fromarray(seg_img)
    save_path ="/data0/jinlinghe/CoLa-Diff/yLabel_cond/sample/888678/maskColor_FA_25.jpg"
    seg_img.save(save_path)




    return seg_img

def calIndex2Dto3D(true_img_path,pred_img_path,mask_path):
    # image1 = Image.open(true_img_path)
    # image2 = Image.open(pred_img_path)
    # true_img_path一个3D脑部nii数据
    # pred_img_path合成的一个脑部数据的50章切片
    eval=[]
    true_img=nib.load(true_img_path)

    true_img = true_img.get_fdata()

    for filename in os.listdir(pred_img_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(pred_img_path, filename)
            # print(file_path)
            #/home/jinlinghe/pythonData/CoLa_Diff_MultiModal_MRI_Synthesis-main/result/step_85000/888678/85000_FA_45.png

            pred_img_sliced = Image.open(file_path)
            slice_index = filename.split(".")[0].split("_")[2]
            # slice_index=filename.split(".")[0].split("_")[2][1:]
            slice_index=int(slice_index)
            print(slice_index)

            true_img_sliced=true_img[:, :, slice_index]
            # pad到相同尺寸
            true_img_sliced = pad_nii(true_img_sliced,(192,192))



            # 将数据转为uint8类型（0-255范围内），因为PIL需要这个数据类型
            true_img_sliced = ((true_img_sliced - true_img_sliced.min()) / (true_img_sliced.max() - true_img_sliced.min()) * 255).astype(np.uint8)
            # 使用PIL保存图像
            true_img_sliced = Image.fromarray(true_img_sliced, 'L')  # 'L'表示灰度图像.resize((256,256))


            eval_sliced = dict()
            # 获取mask
            mask = get_mask(mask_path, slice_index)

            # 边缘去噪
            pred_img_sliced = edg_denoise_img2(pred_img_sliced,mask)
            true_img_sliced = edg_denoise_img2(true_img_sliced,mask)
            eval_sliced["PSNR"],eval_sliced["SSIM"],eval_sliced["NMSE"]=eval_slice(true_img_sliced,pred_img_sliced,mask)

            eval.append(eval_sliced)
    print(eval)

    #求平均值
    psnr_value = np.mean([entry['PSNR'] for entry in eval])
    ssim_value = np.mean([entry['SSIM'] for entry in eval])
    nmse_value = np.mean([entry['NMSE'] for entry in eval])

    print(f"PSNR: {psnr_value}")
    print(f"SSIM: {ssim_value}")
    print(f"NMSE: {nmse_value}")

    return psnr_value, ssim_value, nmse_value  # lpips_value,NIQE_value












    # / data3 / langzhang / dtidata_test / 888678
    #/home/jinlinghe/pythonData/CoLa_Diff_MultiModal_MRI_Synthesis-main/result/step_85000/888678


def edg_denoise_img(image):
#    image=Image.open(img_path)输入为Image.open打开的图片

    # Read the input image
    # image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


    image=np.array(image)


    # print(image.max())
    # print(image.min())
    # #
    # edges = ndimage.sobel(image)
    # image=edges
    # edges = cv2.Canny(image, 20, 700)


    # 假设 image 是你的图像数据
    # 二值化图像（你可以使用适当的阈值或其他二值化方法）
    # 高阈值开操作。边界噪声侵蚀
    open_binary_Ori = (image >  80).astype(np.uint8)


    # 定义核
    kernel_size =5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)


    open_binary_image = binary_opening(open_binary_Ori, structure=kernel, iterations=1)
    # binary_erosion_result=binary_closing(binary_erosion_result, structure=kernel, iterations=1)


    open_noise=open_binary_Ori-open_binary_image







    # noise=noise/ 255.0


    # #去除噪声
    # open_result = np.copy(norm_image)
    # open_result[open_noise==True] = 0

    # 若只进行开操作会，图像中间形成了白噪声点

    #低阈值闭操作
    close_binary_Ori = (image >10).astype(np.uint8)


    # mask = (image > 0).astype(np.uint8)
    close_binary=close_binary_Ori-open_noise
    close_binary_image=binary_closing(close_binary, structure=kernel, iterations=1)
    close_noise = close_binary_Ori - close_binary_image


    # close_noise = close_binary_image - close_binary_Ori


    # open_result = open_result / 255.0
    norm_image = image / 255.0
    close_result=np.copy(norm_image)
    close_result[close_binary_image==False] = 0


    result=close_result
    # result=close_binary_Ori*image


    # 显示原始图像和侵蚀后的图像
#     plt.subplot(2, 4, 1)
#     plt.imshow(image, cmap='gray')
#     plt.title('Original Image')
#     plt.axis('off')
#
#     plt.subplot(2, 4, 2)
#     plt.imshow(open_binary_Ori, cmap='gray')
#     plt.title('Binary Image')
#     plt.axis('off')
#
#     plt.subplot(2, 4, 3)
#     plt.imshow(open_binary_image, cmap='gray')
#     plt.title('Open Result')
#     plt.axis('off')
#
#     plt.subplot(2, 4, 4)
#     plt.imshow(open_noise, cmap='gray')
#     plt.title('noise')
#     plt.axis('off')
# #############################################
#
#     plt.subplot(2, 4, 5)
#     plt.imshow(close_binary_Ori, cmap='gray')
#     plt.title('Mask')
#     plt.axis('off')
#
#
#     plt.subplot(2, 4, 6)
#     plt.imshow(close_binary, cmap='gray')
#     plt.title('Denoising first')
#     plt.axis('off')
#
#     plt.subplot(2, 4, 7)
#     plt.imshow(close_binary_image, cmap='gray')
#     plt.title('Close Result')
#     plt.axis('off')
#
#     plt.subplot(2, 4, 8)
#     plt.imshow(close_result, cmap='gray')
#     plt.title('Denoising Image')
#     plt.axis('off')
#
#     plt.show()
#
#     plt.subplot(1,1, 1)
#     plt.imshow(close_noise, cmap='gray')
#     plt.title('Marginal Noise')
#     plt.axis('off')
#
#     plt.show()

    # save_path = '/home/jinlinghe/pythonData/CoLa_Diff_MultiModal_MRI_Synthesis-main/result/888678/ero_FA_Zori_splited_25'
    # plt.savefig(save_path)

    # 使用PIL保存图像
    result = ((result - result.min()) / (result.max() - result.min()) * 255).astype(np.uint8)
    result = Image.fromarray(result, 'L')  # 'L'表示灰度图像.resize((256,256))
#     result.save("/home/jinlinghe/pythonData/CoLa_Diff_MultiModal_MRI_Synthesis-main/result/888678/85000_FA_25_denoised.png")


    return result


def edg_denoise_img2(image,mask,kernel_size):



    # kernel_size = 4
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    mask = binary_erosion(mask, structure=kernel, iterations=1)
    # #
    # pre_trained_models = eval_slice(true_image,pred_image,mask)
    img = image* mask


    # plt.subplot(2, 1, 1)
    # plt.imshow(image, cmap='gray')
    # plt.title('OriginImg')
    # plt.axis('off')
    # plt.subplot(2, 1, 2)
    # plt.imshow(img, cmap='gray')
    # plt.title('DenoiseImg')
    # plt.axis('off')
    #
    # plt.show()

    # #转换成image图像
    # img= ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    # img = Image.fromarray(img, 'L')
    # img.save("/home/jinlinghe/pythonData/CoLa_Diff_MultiModal_MRI_Synthesis-main/result/888678/85000_FA_40_ero.png")

    return img

def datasets_calIndex():

    return


def calIndex3Dnii(true_img_path, pred_img_path, mask_path,kernel_size):
    eval = []
    true_img = nib.load(true_img_path)
    true_img = true_img.get_fdata()

    pred_img = nib.load(pred_img_path)
    pred_img = pred_img.get_fdata()

    mask=nib.load(mask_path)
    mask=mask.get_fdata()
    for slice_index in range(true_img.shape[-1]):
        true_img_sliced = true_img[..., slice_index]
        pred_img_sliced = pred_img[..., slice_index]
        # 获取mask
        mask_sliced = mask[..., slice_index]
        # pad到相同尺寸
        true_img_sliced = pad_nii(true_img_sliced, (192, 192))
        pred_img_sliced = pad_nii(pred_img_sliced, (192, 192))
        mask_sliced=pad_nii(mask_sliced, (192, 192))
        #MD的值需要截断（test_GT数据未截断）
        true_img_sliced=np.clip(true_img_sliced, 0, 0.0035)
        pred_img_sliced = np.clip(pred_img_sliced, 0, 0.0035)

        eval_sliced = dict()
        # 边缘去噪
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_ero = binary_erosion(mask_sliced, structure=kernel, iterations=1)
        pred_img_sliced=pred_img_sliced*mask_ero
        true_img_sliced=true_img_sliced*mask_ero
        pred_img_sliced = edg_denoise_img2(pred_img_sliced, mask_sliced,kernel_size)
        true_img_sliced = edg_denoise_img2(true_img_sliced, mask_sliced,kernel_size)

        eval_sliced["PSNR"], eval_sliced["SSIM"], eval_sliced["NMSE"] = eval_slice(true_img_sliced, pred_img_sliced,
                                                                                   mask_sliced)


        eval.append(eval_sliced)
    # print(eval)

    # 求平均值
    psnr_value = np.mean([entry['PSNR'] for entry in eval])
    ssim_value = np.mean([entry['SSIM'] for entry in eval])
    nmse_value = np.mean([entry['NMSE'] for entry in eval])

    # print(f"PSNR: {psnr_value}")
    # print(f"SSIM: {ssim_value}")
    # print(f"NMSE: {nmse_value}")

    return psnr_value, ssim_value, nmse_value

def calIndex3Dnii_RGB(true_img_path, pred_img_path, mask_path,kernel_size):
    eval = []
    true_img = nib.load(true_img_path)
    true_img = true_img.get_fdata()

    pred_img = nib.load(pred_img_path)
    pred_img = pred_img.get_fdata()

    mask=nib.load(mask_path)
    mask=mask.get_fdata()

    # for slice_index in range(true_img.shape[-1]):
    for slice_index in range(true_img.shape[-2]):#colorFA
        # print(slice_index)
        true_imgnp_sliced = true_img[..., slice_index,:]
        pred_imgnp_sliced = pred_img[..., slice_index,:]
        # 获取mask
        mask_sliced = mask[..., slice_index]
        mask_sliced = pad_nii(mask_sliced, (192, 192))

        true_img_sliced=np.zeros((192,192,3))
        pred_img_sliced = np.zeros((192, 192, 3))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_ero = binary_erosion(mask_sliced, structure=kernel, iterations=1)
        for channel in range(3):
            # pad到相同尺寸
            true_img_sliced[:, :, channel] = pad_nii(true_imgnp_sliced[:, :, channel],(192,192))
            pred_img_sliced[:, :, channel] = pad_nii(pred_imgnp_sliced[:, :, channel], (192, 192))
            eval_sliced = dict()
            # 边缘去噪
            pred_img_sliced[:, :, channel]=pred_img_sliced[:, :, channel]*mask_ero
            true_img_sliced[:, :, channel]=true_img_sliced[:, :, channel]*mask_ero
            # pred_img_sliced[:, :, channel] = edg_denoise_img2(pred_img_sliced[:, :, channel], mask_sliced,kernel_size)
            # true_img_sliced[:, :, channel] = edg_denoise_img2(true_img_sliced[:, :, channel], mask_sliced,kernel_size)

        # 对于Color_FA,mask扩展一个维度
        mask_sliced=np.expand_dims(mask_sliced,-1)
        eval_sliced["PSNR"], eval_sliced["SSIM"], eval_sliced["NMSE"] = eval_slice(true_img_sliced, pred_img_sliced,
                                                                                   mask_sliced)


        eval.append(eval_sliced)
    # print(eval)

    # 求平均值
    psnr_value = np.mean([entry['PSNR'] for entry in eval])
    ssim_value = np.mean([entry['SSIM'] for entry in eval])
    nmse_value = np.mean([entry['NMSE'] for entry in eval])

    # print(f"PSNR: {psnr_value}")
    # print(f"SSIM: {ssim_value}")
    # print(f"NMSE: {nmse_value}")

    return psnr_value, ssim_value, nmse_value
def show_img(img):
    # min_val = min(img.any())
    # max_val = max(img.any())
    # print(f"Min value : {min_val}")
    # print(f"Max value: {max_val}")

    plt.figure(figsize=(15, 5))
    # plt.imshow(a, cmap='gray')
    plt.imshow(img)#, cmap='Blues'
    plt.colorbar()
    plt.show()
    return 0

def denoising_dti(data, N):
    # 将图像转换为浮点数格式
    data_float = img_as_float(data)

    # 估计噪声标准差
    sigma_est = np.mean(estimate_sigma(data_float, N))
    print(sigma_est)
    # 应用非局部均值去噪
    data_denoised = denoise_nl_means(data_float, h=2.5 * sigma_est, fast_mode=True, patch_size=3, patch_distance=1,
                                     multichannel=True)

    return data_denoised

if __name__ == '__main__':
    ############压缩



    # # # # # #####todo:2D_RGB切片指标评估
    #注意这里true_img是单通道，pred_image是3通道
    true_img_path="/data0/jinlinghe/CoLa-Diff/slices_img/888678/FAori_Z_25.png"
    # true_img_path = "/home/jinlinghe/pythonData/CoLa_Diff_MultiModal_MRI_Synthesis-main/result/step_85000/888678/85000_MD_40.png"
    # # #"/home/jinlinghe/pythonData/CoLa_Diff_MultiModal_MRI_Synthesis-main/result/888678/FA_ZoriB_splited_25.png"
    # # # #
    # # # # pred_img_path="/data0/jinlinghe/CoLa-DiffD/MaskCond/synthesis_img/11000/888678/25000_FA_B25.png"
    pred_img_path="/data0/jinlinghe/CoLa-Diff/yLabel_cond_ad/sample_CBAM_numhead4/888678/23000_FA_25.png"
    mask=get_mask('/data3/langzhang/dtidata_test/888678/mask.nii.gz',25)
    true_image = Image.open(true_img_path)
    pred_image = Image.open(pred_img_path)
    true_image = np.array(true_image)
    pred_image = np.array(pred_image)
    # # pred_image=np.transpose(pred_image,(1,0,2))
    # # true_image= edg_denoise_img2(true_image, mask, 5)
    # # true_image = np.stack((true_image,) * 3, axis=-1)
    # #RGB图按三个方向分别去噪
    # for c in range(3):
    #     # print(true_image[...,c].shape)
    #     true_image[...,c]=edg_denoise_img2(true_image[...,c],mask,5)
    #     pred_image[...,c]=edg_denoise_img2(pred_image[...,c],mask,5)
    # # #计算指标
    # mask=np.stack((mask,) * 3, axis=-1)
    # # show_img(true_image)
    # # show_img(pred_image)
    #
    # # true_image=edg_denoise_img2(true_image,mask,5)
    # # pred_image = edg_denoise_img2(pred_image, mask, 5)
    # psnr_value,ssim_value,nmse_value = eval_slice(true_image,pred_image,mask)
    ###指标显示
    # print(f"PSNR: {psnr_value}")
    # print(f"SSIM: {ssim_value}")
    # print(f"NMSE: {nmse_value}")
    # # # # #todo:2D_grey
    # pred_image=np.transpose(pred_image,(1,0,2))
    # pred_image=pred_image[...,0]
    true_image=edg_denoise_img2(true_image,mask,5)
    pred_image=edg_denoise_img2(pred_image,mask,5)
    # # #计算指标
    psnr_value,ssim_value,nmse_value = eval_slice(true_image,pred_image,mask)
    # ####指标显示
    print(f"PSNR: {psnr_value}")
    print(f"SSIM: {ssim_value}")
    print(f"NMSE: {nmse_value}")






    #
    #
    # kernel_size = 4
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # mask = binary_erosion(mask, structure=kernel, iterations=1)
    # img=true_image*mask
    #
    #
    # plt.show()

    #
    # #
    # # #########todo:3D图片与2D切片指标评估
    # true_img_path="/data3/langzhang/dtidata_test/578057/FA.nii.gz"
    # pred_img_path="/home/jinlinghe/pythonData/CoLa_Diff_MultiModal_MRI_Synthesis-main/result/step_85000/578057/"
    # mask_path='/data3/langzhang/dtidata_test/578057/mask.nii.gz'
    # # mask=get_mask('/data3/langzhang/dtidata_test/888678/mask.nii.gz',25)
    # # true_img
    # # edg_denoise_img2(true_img_path,mask_path)
    # #
    # # /home/jinlinghe/pythonData/CoLa_Diff_MultiModal_MRI_Synthesis-main/result/step_106000/578057
    # calIndex2Dto3D(true_img_path,pred_img_path,mask_path)
    # # # #






    # ############图片侵蚀边界
    # image=Image.open(true_img_path)
    # edg_denoise_img(image)
    # # eval_slice()

    # #mask_seg
    # mask=get_mask('/data3/langzhang/dtidata_test/888678/mask.nii.gz',25)
    # img_path = "/data0/jinlinghe/CoLa-Diff/yLabel_cond/sample/888678/Color_FA25.jpg"
    # img=Image.open(img_path)
    # mask_seg(mask,img)

    # # # #########todo:3Dnii图片指标评估,单个数据
    # true_data_path="/data3/jinlinghe/dtidata_test"
    # pred_data_path="/data3/jinlinghe/deepDTIData_test"
    # numbers=os.listdir(pred_data_path)
    # eval=[]
    # kernel_size = 4
    # #注意MD的值在归一化的时候要先截断
    # for number in numbers:
    #     #跳过非数字文件夹
    #     if number.isdigit() is False:
    #         continue
    #     # if number=="103010" or number=="827052":#异常
    #     #     continue
    #     one_eval=dict()
    #     true_img_path = os.path.join(true_data_path, number, "AD.nii.gz")
    #     pred_img_path = os.path.join(pred_data_path,number,"preDenoise_AD_cnn.nii.gz")
    #     mask_path=os.path.join(true_data_path, number, "mask.nii.gz")
    #     one_psnr,one_ssim,one_nmse=calIndex3Dnii(true_img_path,pred_img_path,mask_path,kernel_size)
    #
    #     # 写入字典one_eval
    #     one_eval["number"]=number
    #     one_eval["PSNR"]=one_psnr
    #     one_eval["SSIM"]=one_ssim
    #     one_eval["NMSE"]=one_nmse
    #
    #     eval.append(one_eval)
    # # 计算总指标并添加到eval列表:
    # psnr_value = np.mean([entry['PSNR'] for entry in eval])
    # ssim_value = np.mean([entry['SSIM'] for entry in eval])
    # nmse_value = np.mean([entry['NMSE'] for entry in eval])
    # mean_eval=dict()
    # mean_eval["number"] = "mean_eval"
    # mean_eval["PSNR"] = psnr_value
    # mean_eval["SSIM"] = ssim_value
    # mean_eval["NMSE"] = nmse_value
    # eval.append(mean_eval)
    # #保存
    # eval_savepath=os.path.join(pred_data_path,f"kernel{kernel_size}_denoise_evalColor_FA.pkl")
    # # 使用 pickle.dump() 函数将字典列表数据保存到文件中
    # with open(eval_savepath, 'wb') as file:
    #     pickle.dump(eval, file)
    # # 加载数据
    # with open(eval_savepath, 'rb') as file:
    #     eval_data = pickle.load(file)
    #
    # # # 打印加载的数据
    # # print("加载的数据:", loaded_data)
    # df = pd.DataFrame(eval_data)
    # # df.to_csv(eval_savepath, sep='\t', index=False)
    # print(df)
