
import numpy as np
import pydensecrf.densecrf as dcrf
from skimage import io
import tifffile as tiff


def crf_postprocessing(image, prob_map, num_classes, sxy_gaussian=3, compat_gaussian=3, sxy_bilateral=50,
                       srgb_bilateral=13, compat_bilateral=10):
    """
    使用CRFs对分类结果进行后处理。
    参数:
    - image: 原始影像（H, W, 3）。
    - prob_map: 初步分类概率图（H, W, num_classes）。
    - num_classes: 分类类别总数。
    - sxy_gaussian, compat_gaussian: 高斯核的空间和相容性参数。
    - sxy_bilateral, srgb_bilateral, compat_bilateral: 双边核的空间、RGB和相容性参数。

    返回:
    - 优化后的标签图（H, W）。
    """
    h, w, _ = image.shape  # 获取图像的高度和宽度
    d = dcrf.DenseCRF2D(h, w, num_classes)  # 创建CRF对象

    # 修复概率图，确保其值在 [1e-6, 1] 范围内
    prob_map = np.clip(prob_map, 1e-6, 1.0)

    # 将分类结果的概率图设置为置信项（局部项）
    prob_map = prob_map.transpose(2, 0, 1)  # 转换为 (num_classes, H, W)
    unary = -np.log(prob_map)  # 负对数概率
    unary = unary.reshape((num_classes, -1))  # 转为 (num_classes, H * W)
    unary = np.ascontiguousarray(unary)  # 确保是 C-contiguous 内存布局

    # 设置 CRF 的局部能量
    d.setUnaryEnergy(unary)

    # 添加高斯对称项（邻域平滑项）
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # **添加双边核对称项（考虑影像颜色相似性），确保传递所有必需参数**
    d.addPairwiseBilateral(
        sxy=sxy_bilateral,
        srgb=srgb_bilateral,
        rgbim=image,  # 灾后影像
        compat=compat_bilateral,
        kernel=dcrf.DIAG_KERNEL,  # 传递内核类型
        normalization=dcrf.NORMALIZE_SYMMETRIC  # 选择适合的归一化方式
    )
    # d.addPairwiseBilateral(sxy=80, srgb=15, rgbim=image, compat=10)

    # 执行推理得到优化后的标签
    Q = d.inference(10)  # 运行10次迭代
    result = np.argmax(Q, axis=0).reshape(h, w)  # 返回优化后的像素标签图
    return result


if __name__ == "__main__":
    # 加载影像和分类概率图
    image_path = "/data01/YR/building/crfs/image_path/_0_0_before.png"  # 灾后影像路径
    prob_map_path = "/data01/YR/building/crfs/prob_map_path/0_original.tif"  # 分类概率图路径 (.npy格式)
    image = io.imread(image_path)  # 读取灾后影像（假设已预处理为 HxWx3 的RGB图像）
    # prob_map = np.load(prob_map_path)  # 加载分类概率图 (HxWxN 的numpy数组，其中N=类别数)
    prob_map = tiff.imread(prob_map_path).transpose(1, 2, 0)
    # # 最小值
    # min_val = np.min(prob_map)
    # # 最大值
    # max_val = np.max(prob_map)
    # # 归一化到 [0, 1]
    # prob_map = (prob_map - min_val) / (max_val - min_val)
    # 确定分类类别数
    num_classes = prob_map.shape[-1]

    # 使用CRFs进行后处理
    refined_labels = crf_postprocessing(image, prob_map, num_classes)
    tiff.imwrite("/data01/YR/building/crfs/results/refined_labels.tif", refined_labels.astype(np.float32))
    # 保存优化后的标签图
    io.imsave("/data01/YR/building/crfs/results/refined_labels.png", (refined_labels * (255 // num_classes)).astype(np.uint8))
    print("CRF后处理完成，结果已保存至 'refined_labels.png'")
