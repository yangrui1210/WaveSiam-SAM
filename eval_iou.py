# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import distance_transform_edt as bwdist

import cv2
_EPS = np.spacing(1)    # the different implementation of epsilon (extreme min value) between numpy and matlab
_TYPE = np.float64


from PIL import Image
from pathlib import Path
import numpy as np
import torch 
#from tqdm import tqdm
#from tqdm.contrib import tzip
#from tqdm import trange
import tqdm
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
__all__ = ['SegmentationMetric']
 
"""
confusionMetric
L\P     P    N
 
P      TP    FN
 
N      FP    TN
 
"""
class SegmentationMetric(object):
    def __init__(self, numClass,ignore_bg):
        self.numClass = numClass
        self.ignore_bg = ignore_bg
        if self.ignore_bg :
            self.confusionMatrix = np.zeros((self.numClass-1,)*2)
        else:
            self.confusionMatrix = np.zeros((self.numClass,)*2)
 
    def overallAccuracy(self):
        # return all class overall pixel accuracy,AO评价指标
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc
  
    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=0) + np.sum(self.confusionMatrix, axis=1) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        #print('IOU:', IoU)
        mIoU = np.nanmean(IoU)
        return mIoU,IoU

    def precision(self):
        #precision = TP / TP + FP
        p = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        return p
    
    def recall(self):
        #recall = TP / TP + FN
        r = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return r
 
    # def genConfusionMatrix(self, imgPredict, imgLabel):
    #     # remove classes from unlabeled pixels in gt image and predict
    #     mask = (imgLabel >= 0) & (imgLabel < self.numClass)#过滤掉其它类别
    #     label = self.numClass * imgLabel[mask] + imgPredict[mask]
    #     count = np.bincount(label, minlength=self.numClass**2)
    #     confusionMatrix = count.reshape(self.numClass, self.numClass)
    #     return confusionMatrix
    

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        #print(mask)
        #print(mask.shape())
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)#[:self.numClass**2]
        confusionMatrix = count.reshape(self.numClass, self.numClass)

        if self.ignore_bg:
            return confusionMatrix[:self.numClass-1, :self.numClass-1]
        else:
            return confusionMatrix
    
    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
 
    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        #print(imgPredict.shape)
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
 
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))
    

def color_to_list(mask, palette=[ [1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,0,0]] ):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = np.zeros([1024,1024],dtype=np.int8)
    for i,colour in enumerate( palette):
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map += class_map*int(i)

    return semantic_map
 
if __name__ == '__main__':   

    true_path = '/data01/YR/building/xBD/hold-test2/targets-post'
    pred_path = '/data01/YR/building/xBD/hold-test2/after_class_SAM_DWT'

    class_num = 5
    metric = SegmentationMetric(class_num, True)
    true_list = Path(true_path).rglob('*')
    pred_list = Path(pred_path).rglob('*')
    img_len = len(os.listdir(true_path))

    for true, pred in tqdm.tqdm(zip(true_list, pred_list), total=img_len):
        true_img = np.array(Image.open(true), dtype=np.uint8).flatten()
        tmp1 = true_img.copy()
        tmp1[tmp1 == 0] = 10
        tmp1[tmp1 == 4] = 0
        tmp1[tmp1 == 10] = 4

        pred_img = np.array(Image.open(pred), dtype=np.uint8).flatten()
        tmp2 = pred_img.copy()
        tmp2[tmp2 == 0] = 10
        tmp2[tmp2 == 4] = 0
        tmp2[tmp2 == 10] = 4


        metric.addBatch(tmp2, tmp1)

    oa = metric.overallAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    p = metric.precision()
    mp = np.nanmean(p)
    r = metric.recall()
    mr = np.nanmean(r)
    f1 = (2 * p * r) / (p + r)
    mf1 = np.nanmean(f1)
    FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
    normed_confusionMatrix = metric.confusionMatrix / metric.confusionMatrix.sum(axis=0)
    normed_confusionMatrix = np.around(normed_confusionMatrix, decimals=3)
    # print('total pixels:', metric.confusionMatrix.sum())
    # print('1024*1024*80=',1024*1024*80)

    axis_labels = ['building', 'vegetation', 'water', 'road', 'background']
    plt.figure()  # figsize=(8, 8))
    sns.heatmap(normed_confusionMatrix, annot=True, cmap='Blues', yticklabels=axis_labels, xticklabels=axis_labels)

    # plt.ylim(0, 4)

    plt.ylabel('Predicted labels')
    plt.xlabel('True labels')
    # plt.yticks(np.array(range(0,5)), axis_labels)
    plt.savefig(true_path.split('/gt')[0] + '/confusionmatrix.jpg')
    # print('self.confusionMatrix:',metric.confusionMatrix / metric.confusionMatrix.sum(axis=0))
    print('self.confusionMatrix:', normed_confusionMatrix)

    print(f' 类别0,类别1,...\n oa:{oa}, \n mIou:{mIoU}, \n p:{p}, \n mp:{mp},  \n r:{r}, \n mr:{mr}, \n f1:{f1}, \n mf1:{mf1}')