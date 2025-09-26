import numpy as np
from scipy.interpolate import interp2d
from PIL import Image
import os
import tifffile as tiff
import segment_before as sam
import numpy as np
import rasterio
from scipy.ndimage import zoom
from coor import Coor
import shutil
from entropy import entropyWeightsScore
from collections import Counter
from tqdm import tqdm


before_path = "/home/wl/gd/00/may/tif/"
dirs = os.listdir(before_path)
before_png = "/data01/YR/earthquake/00/volcano/pre/png"
# before_tif = "/data01/YR/20250107dingri/datasets/after_tif"
sam_tif = "/home/wl/gd/00/may/sam"

before_tif_path = "/data01/YR/building/paper-data/images-pre-tif"

# # 仅分割
# for data in os.listdir(before_path):
#     image = Image.open(os.path.join(before_path, data))
#     tiff.imwrite(os.path.join(before_tif_path, data), image)
#     sam.samGeo(os.path.join(before_tif_path, data),
#                os.path.join(sam_tif, data)).auto_generate_mask()

# # 影像补全
# for dir in dirs:
#     print("================== start process file: " + dir + " ==================")
#     # 读取TIFF图像，提取前三个波段
#     # data = tiff.imread(os.path.join(before_path, dir))[:, :, 1:]
#     data = tiff.imread(os.path.join(before_path, dir))
#     # 提取所需的波段
#     nir_band = data[:, :, 0]  # 波段 5 (NIR)
#     red_band = data[:, :, 1]  # 波段 4 (Red)
#     green_band = data[:, :, 2]  # 波段 3 (Green)
#     # 将波段数据堆叠成一个 RGB 影像
#     data = np.dstack((nir_band, red_band, green_band))
#     # 获取原图的尺寸
#     height, width, _ = data.shape
#     # 将行列不足512的部分用0填充
#     new_width = max(width, 1024)
#     new_height = max(height, 1024)
#     # 创建一个全黑的图片，大小为512x512（或者更大）
#     padded_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
#     # 将NumPy数组转换为PIL图像，并粘贴到黑色背景上
#     pil_image = Image.fromarray(data.astype(np.uint8))
#     padded_image.paste(pil_image, (0, 0))
#     # 将行列翻倍：可以通过插值的方式来缩放图片，使用BICUBIC插值方式进行高质量的缩放
#     # doubled_image = padded_image.resize((new_width * 2, new_height * 2), Image.BICUBIC)
#     doubled_image = padded_image
#     # 保存TIFF图像
#     tiff.imwrite(os.path.join(before_path, dir), np.array(doubled_image))  # 将PIL图像转换为NumPy数组并保存为TIFF
#     # 保存为PNG格式
#     # doubled_image.save(os.path.join(before_png, dir.split('.')[0] + '.png'))


# # 影像补全与分割
# for dir in dirs:
#     print("================== start process file: " + dir + " ==================")
#     # 读取TIFF图像，提取前三个波段
#     # data = tiff.imread(os.path.join(before_path, dir))[:, :, 1:]
#     data = tiff.imread(os.path.join(before_path, dir))
#     # 提取所需的波段
#     nir_band = data[:, :, 0]  # 波段 5 (NIR)
#     red_band = data[:, :, 1]  # 波段 4 (Red)
#     green_band = data[:, :, 2]  # 波段 3 (Green)
#     # 将波段数据堆叠成一个 RGB 影像
#     data = np.dstack((nir_band, red_band, green_band))
#     # 获取原图的尺寸
#     height, width, _ = data.shape
#     # 将行列不足512的部分用0填充
#     new_width = max(width, 1024)
#     new_height = max(height, 1024)
#     # 创建一个全黑的图片，大小为512x512（或者更大）
#     padded_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
#     # 将NumPy数组转换为PIL图像，并粘贴到黑色背景上
#     pil_image = Image.fromarray(data.astype(np.uint8))
#     padded_image.paste(pil_image, (0, 0))
#     # 将行列翻倍：可以通过插值的方式来缩放图片，使用BICUBIC插值方式进行高质量的缩放
#     # doubled_image = padded_image.resize((new_width * 2, new_height * 2), Image.BICUBIC)
#     doubled_image = padded_image
#     # 保存TIFF图像
#     tiff.imwrite(os.path.join(before_path, dir), np.array(doubled_image))  # 将PIL图像转换为NumPy数组并保存为TIFF
#     if np.all(data == 0):
#         print("data==0")
#         shutil.copy(os.path.join(before_path, dir), os.path.join(sam_tif, dir))
#     else:
#         sam.samGeo(os.path.join(before_path, dir),
#                 os.path.join(sam_tif, dir)).auto_generate_mask()
#     # 保存为PNG格式
#     # doubled_image.save(os.path.join(before_png, dir.split('.')[0] + '.png'))


# 分割与建筑物预测结果融合
# sam_path = "/data01/YR/earthquake/AI Bayda/sam"
# predict_path = "/data01/YR/earthquake/AI Bayda/building_predict"
# dirs = sorted(os.listdir(sam_path))+
# for i in range(len(dirs)):
#     print("======================= start files: ", dirs[i])
#     sam = tiff.imread(os.path.join(sam_path, dirs[i]))
#     predict = tiff.imread(os.path.join(predict_path, str(i) + "_original.tif"))[0, :, :]
#     tmp_predict = np.zeros_like(sam)
#     tmp_pre_sam = np.zeros_like(sam)
#     sam_unique = np.unique(sam)
#     # 确保sam和predict形状一致
#     if sam.shape != predict.shape:
#         print(sam.shape, predict.shape)
#         print(f"Warning: Shape mismatch for {dirs[i]}. Skipping this file.")
#         sam = np.zeros_like(sam[:, :, 0])
#         print(sam.shape)
#     for j in sam_unique:
#         # 获取sam中等于j的索引位置
#         indices = np.where(sam == j)
#         if len(indices[0]) > 0:  # 确保indices有有效的位置
#             # 获取这些位置在predict中的值，并计算平均值
#             avg_predict = np.mean(predict[indices])
#             # 根据平均值判断条件并更新tmp_pre_sam
#             if avg_predict > -10:
#                 tmp_pre_sam[indices] = j + 1  # 更新建筑物预测结果
#             tmp_predict[indices] = avg_predict  # 更新预测值
#         else:
#             tmp_predict[indices] = predict[indices]
#     tiff.imwrite("/data01/YR/earthquake/AI Bayda/predict/sam/" + dirs[i], tmp_predict)
#     tiff.imwrite("/data01/YR/earthquake/AI Bayda/predict/building/" + dirs[i], tmp_pre_sam)


# 影像重采样
# 读取影像a和b
a_path = '/data01/YR/DSFA/river/0807/GFQ1.tif'
b_path = '/data01/YR/DSFA/river/0807/HJH.tif'

# 打开影像a
with rasterio.open(a_path) as src_a:
    a_data = src_a.read()  # 读取所有波段的数据，a_data的形状是(3, height, width)
    # a_height, a_width = a_data.shape[1], a_data.shape[2]  # 获取影像a的高度和宽度（行列数）
    a_height, a_width = a_data.shape[1], a_data.shape[2]  # 获取影像a的高度和宽度（行列数）

# 打开影像b
with rasterio.open(b_path) as src_b:
    b_data = src_b.read()  # 读取所有波段的数据，b_data的形状是(4, height, width)

# 提取b的前三个波段
# b_first_three_bands = b_data[:3, :, :]  # 选取前三个波段
b_first_three_bands = b_data[:, :]  # 选取前三个波段

# 2. 重采样影像b的前三个波段以匹配影像a的行列数
# b_resized = np.zeros((3, a_height, a_width), dtype=np.uint8)  # 创建一个新的空数组来存储重采样后的数据
b_resized = np.zeros((a_height, a_width), dtype=np.uint8)  # 创建一个新的空数组来存储重采样后的数据
for i in range(1):  # 仅重采样前三个波段
    b_resized[i] = zoom(b_first_three_bands[i], (a_height / b_first_three_bands.shape[1], a_width / b_first_three_bands.shape[2]), order=3)

# 3. 保存重采样后的影像b（前三个波段）
output_path = '/data01/YR/DSFA/river/0807/HJH1.tif'  # 输出路径
with rasterio.open(output_path, 'w', driver='GTiff', count=1, dtype='uint8', width=a_width, height=a_height,
                   crs=src_a.crs, transform=src_a.transform) as dst:
    dst.write(b_resized)  # 将重采样后的前三个波段写入新的tif文件

print(f"Resampling complete. The resampled image is saved at {output_path}.")

# 读取灾后裁剪影像，并转换成png
# tif_path = "/data01/YR/earthquake/Min Kun/clip/before/tif"
# png_path = "/data01/YR/earthquake/Min Kun/clip/before/png"
# dirs = os.listdir(tif_path)
# for dir in dirs:
#     file_name = dir.split('.')[0]
#     image = tiff.imread(os.path.join(tif_path, dir)).astype(np.uint8)
#     Image.fromarray(image).save(os.path.join(png_path, file_name+".png"))

# # 损毁等级与sam融合
# building_sam_path = '/data01/YR/earthquake/Min Kun/predict/building'
# hot_path = '/data01/YR/earthquake/Min Kun/mask'
# dirs = sorted(os.listdir(building_sam_path))
# after_mean = "/data01/YR/earthquake/Min Kun/after_mean"
# after_class =  "/data01/YR/earthquake/Min Kun/after_class"
# building_num = 0
# class_num = [0, 0, 0, 0]
#
# for i in range(len(dirs)):
#     sam = tiff.imread(os.path.join(building_sam_path, dirs[i]))  # 读取 SAM 图像
#     hot = tiff.imread(os.path.join(hot_path, str(i) + '.tif'))
#
#     # 初始化临时数组，用来保存平均值
#     tmp_class = np.zeros_like(sam)  # 维度与 sam 相同
#
#     sam_unique = np.unique(sam)  # 获取 SAM 图像中所有的类别
#     # print(len(sam_unique))
#
#     for j in sam_unique:
#         if j != 0:  # 排除背景（假设 0 为背景）
#             mask = (sam == j)
#             if len(mask) > 40:
#                 building_num = building_num + 1
#             max_value = np.max(hot[mask])
#             class_num[max_value-1] = class_num[max_value-1]+1
#             if max_value == 0:
#                 tmp_class[mask] = 1
#             else:
#                 tmp_class[mask] = max_value
#
#     # 对 tmp_mean 数组进行平均，沿着 x 和 y 方向（即对二维进行求平均）
#     # tmp_mean = np.mean(tmp_mean, axis=(0, 1))
#     # tmp_class = np.argmax(tmp_mean[:, :, 1:])+1
#     tiff.imwrite(os.path.join(after_class, dirs[i]), tmp_class)
#     print(f"Processed {dirs[i]}, mean values for hot array")
# print("building num: ", building_num)
# print("class num: ", class_num)

# # 损毁等级与sam融合
# building_sam_path = '/data01/YR/earthquake/00/wildfire/sam'
# hot_path = '/data01/YR/earthquake/00/wildfire/mask'
# dirs = sorted(os.listdir(building_sam_path))
# after_class =  "/data01/YR/earthquake/00/wildfire/object"
# building_num = 0
# class_num = [0, 0, 0, 0]
#
# for i in tqdm(range(len(dirs)), ncols=100):
#     # print("start process: ", dirs[i])
#     sam = tiff.imread(os.path.join(building_sam_path, dirs[i]))  # 读取 SAM 图像
#     hot = tiff.imread(os.path.join(hot_path, str(i) + '.tif'))
#     # file_name = '_'.join(dirs[i].split('_')[:2])
#     # hot = tiff.imread(os.path.join(hot_path, file_name + '_post_disaster.tif'))
#     # 初始化临时数组，用来保存平均值
#     tmp_class = np.zeros_like(sam)  # 维度与 sam 相同
#     sam_unique = np.unique(sam)  # 获取 SAM 图像中所有的类别
#     # print(len(sam_unique))
#     for j in sam_unique:
#         # 若第j个对象所有的值为0，则pass掉，不进行后面的计算
#         mask = (sam == j)
#         if len(mask.shape) > 2:
#             pass
#         elif np.all(hot[mask[:, :]] == 0):
#             pass
#         else:
#             count = [Counter(hot[mask])[0], Counter(hot[mask])[1], Counter(hot[mask])[2], Counter(hot[mask])[3], Counter(hot[mask])[4]]
#             damage_level = entropyWeightsScore(count[0], count[1], count[2], count[3], count[4]).calculate_weight_score()
#             if damage_level == 0:
#                 tmp_class[mask] = 0
#             else:
#                 tmp_class[mask] = damage_level
#
#         # if j != 0:  # 排除背景（假设 0 为背景）
#         #     mask = (sam == j)
#         #     if len(mask) > 40:
#         #         building_num = building_num + 1
#         #     max_value = np.max(hot[mask])
#         #     class_num[max_value-1] = class_num[max_value-1]+1
#         #     if max_value == 0:
#         #         tmp_class[mask] = 1
#         #     else:
#         #         tmp_class[mask] = max_value
#     # 对 tmp_mean 数组进行平均，沿着 x 和 y 方向（即对二维进行求平均）
#     # tmp_mean = np.mean(tmp_mean, axis=(0, 1))
#     # tmp_class = np.argmax(tmp_mean[:, :, 1:])+1
#     tiff.imwrite(os.path.join(after_class, dirs[i]), tmp_class)
#     # print(f"Processed {dirs[i]}, mean values for hot array")
# # print("building num: ", building_num)
# # print("class num: ", class_num)

# 拼接
# path1 = "/data01/YR/earthquake/00/wildfire/pre/tif"
# path2 = "/data01/YR/earthquake/00/wildfire/mask"
# path3 = "/data01/YR/earthquake/00/wildfire/mask1"
# dirs = sorted(os.listdir(path1))
# for i in range(len(dirs)):
#     file_name = dirs[i]
#     src = os.path.join(path2, str(i) + ".tif")
#     dst = os.path.join(path3, dirs[i])
#     shutil.copy(src, dst)

# original_tif = "/home/wl/gd/may1.tif"
# ISZ = 1024
# input_path = "/home/wl/gd/00/may/sam"
# output_path = "/home/wl/gd/pinjiemay.tif"
# output_coor_path = "/home/wl/gd/pinjiemay_coor.tif"
# original_image = tiff.imread(original_tif)
# rows = (original_image.shape[0] + ISZ - 1) // ISZ
# cols = (original_image.shape[1] + ISZ - 1) // ISZ
# merge = np.zeros_like(original_image)
# tmp = np.zeros((rows*ISZ, cols*ISZ))
# max_value = 0
# for i in range(rows):
#     for j in range(cols):
#         # classify = tiff.imread(os.path.join(input_path, str(i * cols + j) + ".tif"))
#         classify = tiff.imread(os.path.join(input_path, "_" + str(i) + "_" + str(j) + ".tif"))
#         # max_value = max_value + np.argmax(classify)
#         if len(classify.shape) == 3:
#             classify = classify[:, :, 0]
#         tmp[i*ISZ:(i+1)*ISZ, j*ISZ:(j+1)*ISZ] = classify + max_value
# merge = tmp[:original_image.shape[0], :original_image.shape[1]]
# tiff.imwrite(output_path, merge)
# Coor(original_tif, output_coor_path, output_path).array2raster()
# print(cols)


# Coor("/data01/YR/DSFA/river/0807/zaihou.TIF", "/data01/YR/DSFA/results/00/DSFAOTSUcoor.tif", "/data01/YR/DSFA/results/00/DSFAOTSU.tif").array2raster()


# target_path = "/data01/YR/building/paper-data/targets-post"
# mask_path = "/data01/YR/building/paper-data/mask-DWT"
# dirs = sorted(os.listdir(target_path))
# for i in range(len(dirs)):
#     # file_name = dirs[i].split('_target.png')[0]
#     # os.rename(os.path.join(mask_path, str(i) + '.tif'), os.path.join(mask_path, file_name + '.tif'))
#     target = Image.open(os.path.join(target_path, dirs[i]))
#     target = np.unique(target)
#     if all(x in target for x in [1, 2, 3, 4]):
#         print(dirs[i])








