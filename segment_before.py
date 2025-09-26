

import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import tifffile as tiff
import time
import os


class samGeo:
    def __init__(self,
                 segImagePath,
                 # coorPath,
                 noCoorPath,
                 points_per_side=128,
                 pred_iou_thresh=0.86,
                 stability_score_thresh=0.92,
                 crop_n_layers=1,
                 crop_n_points_downscale_factor=2,
                 min_mask_region_area=20):
        self.segImagePath = segImagePath
        # self.coorPath = coorPath
        self.noCoorPath = noCoorPath
        self.points_per_side = points_per_side
        self.pred_iou_thresh =pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area


    def save_mask(self, anns):
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        mask = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1])).astype(np.int32)
        label = 1
        for ann in sorted_anns:
            m = ann['segmentation']
            mask[m] = label
            label = label+1
        tiff.imwrite(self.noCoorPath, mask)
        # Coor(self.segImagePath, self.coorPath, self.noCoorPath).array2raster()


    def auto_generate_mask(self):
        start_time = time.time()  # 记录开始时间
        print("start time " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        sam_checkpoint = "/data01/YR/building/code/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        torch.cuda.set_device(1)
        device = "cuda"
        image = tiff.imread(self.segImagePath).astype(np.uint8)[:, :, :3]
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=self.points_per_side,  # 控制采样点的间隔，值越小，采样点越密集
            pred_iou_thresh=self.pred_iou_thresh,  # mask的iou阈值
            stability_score_thresh=self.stability_score_thresh,  # mask的稳定性阈值
            crop_n_layers=self.crop_n_layers,
            crop_n_points_downscale_factor=self.crop_n_points_downscale_factor,
            min_mask_region_area=self.min_mask_region_area,  # 最小mask面积，会使用opencv滤除掉小面积的区域
        )
        masks = mask_generator.generate(image)

        print("segment " + str(len(masks)) + " objects.")
        self.save_mask(masks)
        end_time = time.time()  # 记录结束时间
        print("end time " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print("spend time ", end_time - start_time)

def predict_img(predict_img_path, output_path, step):
    # # print("============ start predict ============")
    img = tiff.imread(predict_img_path)
    #
    n = img.shape[0]//step
    c = img.shape[1]//step
    cnv = img
    # 将大图像分割成小图像
    for i in range(n+1):
        for j in range(c+1):
            if i!=n and j!=c:
                tiff.imwrite(output_path + str(i) + '-' + str(j) + '_sam.tif',
                            cnv[i * step:(i + 1) * step, j * step:(j + 1) * step, :].astype(np.uint8))
            elif i == n and j != c:
                tiff.imwrite(output_path + str(i) + '-' + str(j) + '_sam.tif',
                            cnv[i * step:img.shape[0], j * step:(j + 1) * step, :].astype(np.uint8))
            elif i!=n and j == c:
                tiff.imwrite(output_path + str(i) + '-' + str(j) + '_sam.tif',
                            cnv[i * step:(i + 1) * step, j * step:img.shape[1], :].astype(np.uint8))
            elif i==n and j == c:
                tiff.imwrite(output_path + str(i) + '-' + str(j) + '_sam.tif',
                            cnv[i * step:img.shape[0], j * step:img.shape[1], :].astype(np.uint8))
    dirs = os.listdir(output_path)
    # 小图像sam分割
    for i in range(0, len(os.listdir(output_path))):
        print("============================== start process " + dirs[i] + " ==============================")
        if not(np.any(tiff.imread(os.path.join(output_path, dirs[i])))):
            print("the image all elements are 0, pass ......")
        else:
            print("start segment the image ......")
            samGeo(os.path.join(output_path, dirs[i]),
                   os.path.join(output_path, dirs[i])).auto_generate_mask()




