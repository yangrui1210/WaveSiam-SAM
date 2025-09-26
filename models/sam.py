import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import pytorch_wavelets.dwt.lowlevel as lowlevel

from models import register
from .mmseg.models.sam import ImageEncoderViT, MaskDecoder, TwoWayTransformer

logger = logging.getLogger(__name__)
from .iou_loss import IOU
from typing import Any, Optional, Tuple


def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    mask = mask.permute(1,2,0)
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    x=x.permute(2,0,1)
    return x

def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        # print(layer)
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)

class DWTForward(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        """
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]

        # Prepare the filters
        filts = lowlevel.prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_col', filts[0])
        self.register_buffer('h1_col', filts[1])
        self.register_buffer('h0_row', filts[2])
        self.register_buffer('h1_row', filts[3])
        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        yh = []
        ll = x
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel transform
        for j in range(self.J):
            # Do 1 level of the transform
            ll, high = lowlevel.AFB2D.apply(
                ll, self.h0_col, self.h1_col, self.h0_row, self.h1_row, mode)
            yh.append(high)

        return ll, yh

class FMS(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FMS, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*3, in_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(inplace=True),
                                    )
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x,imagename=None):

        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        yH = self.conv_bn_relu(yH)
        yL = self.outconv_bn_relu_L(yL)
        yH = self.outconv_bn_relu_H(yH)
        return yL,yH

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        # 卷积层，用于生成注意力权重
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1)  # 输出通道为1，生成一个权重图

    def forward(self, x):
        # 通过卷积层生成权重图，并用sigmoid归一化到0到1之间
        attention = F.relu(self.conv1(x))
        attention = self.conv2(attention)
        attention = torch.sigmoid(attention)  # 使得注意力值在 [0, 1] 之间
        return attention

class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss

def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # 3x3卷积，用于生成空间注意力图
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算每个空间位置的平均池化和最大池化
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        # 将池化后的特征拼接在一起
        spatial_attention = torch.cat([avg_pool, max_pool], dim=1)
        # 通过卷积生成空间注意力图
        spatial_attention = self.conv(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)
        # 根据空间注意力图加权特征图
        return x * spatial_attention

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # 使用全局平均池化（Global Average Pooling）
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全连接层用于学习通道重要性
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 对输入特征图进行全局平均池化，得到每个通道的平均值
        avg_pool = self.global_avg_pool(x).view(x.size(0), -1)
        # 通过全连接层计算通道的权重
        channel_weights = self.fc1(avg_pool)
        channel_weights = self.relu(channel_weights)
        channel_weights = self.fc2(channel_weights)
        channel_weights = self.sigmoid(channel_weights).view(x.size(0), -1, 1, 1)
        # 按照计算的通道权重调整特征图
        return x * channel_weights

class AttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(AttentionFusion, self).__init__()
        # 通道注意力机制
        self.channel_attention = ChannelAttention(in_channels)
        # 空间注意力机制
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # 首先应用通道注意力机制
        x = self.channel_attention(x)
        # 然后应用空间注意力机制
        x = self.spatial_attention(x)
        return x


@register('sam')
class SAM(nn.Module):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None,num_classes = None, loss_weight = None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = encoder_mode['embed_dim']
        self.image_encoder = ImageEncoderViT(
            img_size=inp_size,
            patch_size=encoder_mode['patch_size'],
            in_chans=3,
            embed_dim=encoder_mode['embed_dim'],
            depth=encoder_mode['depth'],
            num_heads=encoder_mode['num_heads'],
            mlp_ratio=encoder_mode['mlp_ratio'],
            out_chans=encoder_mode['out_chans'],
            qkv_bias=encoder_mode['qkv_bias'],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_rel_pos=encoder_mode['use_rel_pos'],
            rel_pos_zero_init=True,
            window_size=encoder_mode['window_size'],
            global_attn_indexes=encoder_mode['global_attn_indexes'],
        )
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            num_classes = num_classes
        )

        # 第一次卷积：6x1024x1024 -> 64x512x512
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()

        # 第二次卷积：64x512x512 -> 128x256x256
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()

        # 第三次卷积：128x256x256 -> 256x128x128
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.fuseFeature1 = FMS(in_ch=256, out_ch=256)
        self.fuseFeature2 = FMS(in_ch=256, out_ch=256)
        self.attention_fusion = AttentionFusion(1024)
        self.conv_attention = nn.Conv2d(1024, 256, kernel_size=1)
        # # 注意力模块，分别用于不同的特征
        # self.attention_L1 = AttentionModule(in_channels=256)  # 假设输入通道数为256
        # self.attention_H1 = AttentionModule(in_channels=256)
        # self.attention_L2 = AttentionModule(in_channels=256)
        # self.attention_H2 = AttentionModule(in_channels=256)

        if 'evp' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if "prompt" not in k and "mask_decoder" not in k and "prompt_encoder" not in k:
                    p.requires_grad = False
        self.loss_mode = loss
        if self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
        elif self.loss_mode == 'bbce':
            self.criterionBCE = BBCEWithLogitLoss()
        elif self.loss_mode == 'iou':
            #self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            #pos_weight = torch.tensor([1.5, 1, 0.5, 1.9, 0.1], dtype=torch.float)
            if loss_weight is not None:
                pos_weight = torch.tensor(loss_weight,dtype=torch.float)
                self.criterionBCE =  torch.nn.CrossEntropyLoss(pos_weight)
            else:
                self.criterionBCE =  torch.nn.CrossEntropyLoss()
            
            self.criterionIOU = IOU()

        # elif self.loss_mode == 'iou_ce':
        #     self.criterionBCE =  torch.nn.CrossEntropyLoss()
        #     self.criterionIOU = IOU()

        self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])

    def set_input(self, input, gt_mask):
        self.input = input.to(self.device)
        self.gt_mask = gt_mask.to(self.device)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)


    def to_image(self, feature_tensor):
        from sklearn.decomposition import PCA
        from scipy.ndimage import zoom
        # 将PyTorch张量转换为NumPy数组以进行PCA
        feature_numpy = feature_tensor.cpu().numpy().reshape(256, -1).T  # 转换并重塑
        # 使用PCA进行降维处理
        pca = PCA(n_components=3)
        feature_reduced = pca.fit_transform(feature_numpy)
        # 将降维后的数据 reshape 回三维
        feature_reduced_numpy = feature_reduced.T.reshape(3, 64, 64)
        # 插值重构到 (1024, 1024) 使用每个降维后的通道
        interpolated_layers = []
        for i in range(feature_reduced_numpy.shape[0]):
            interpolated_layer = zoom(feature_reduced_numpy[i], (1024 / 64, 1024 / 64), order=3)  # 使用三次插值
            interpolated_layers.append(interpolated_layer)
        # 将插值后的数据再转换为PyTorch张量
        interpolated_map_tensor = torch.tensor(interpolated_layers)
        # 聚合同一张图展示多个通道 (合并为一个通道用于展示)
        aggregated_map = torch.mean(interpolated_map_tensor, dim=0).numpy()
        return aggregated_map


    def forward(self):
        bs = 1

        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        # self.features = self.image_encoder(self.input)
        c = self.input.shape[1] // 2
        input1 = self.input[:, :c, :, :]  # 提取灾前数据
        input2 = self.input[:, c:, :, :]  # 提取灾后数据

        # 前向传播：经过三次卷积层
        x = self.conv1(self.input)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        fusefeature_L1, fusefeature_H1 = self.fuseFeature1(x, None)

        self.features1 = self.image_encoder(input1)
        self.features2 = self.image_encoder(input2)

        # self.features = self.features1 + self.features2

        fused_features = torch.cat([self.features1, self.features2, fusefeature_L1, fusefeature_H1], dim=1)
        # 应用注意力机制进行加权
        fused_features = self.attention_fusion(fused_features)
        # 通过1x1卷积调整通道数
        self.features = self.conv_attention(fused_features)

        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        self.pred_mask = masks

    def infer(self, input):
        bs = 1

        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        # self.features = self.image_encoder(input)  #第一个val 第二张图推理循环 显存+5G

        c = input.shape[1] // 2
        input1 = input[:, :c, :, :]  # 提取灾前数据
        input2 = input[:, c:, :, :]  # 提取灾后数据

        # 前向传播：经过三次卷积层
        x = self.conv1(input)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        fusefeature_L1, fusefeature_H1 = self.fuseFeature1(x, None)
        fusefeature_L1_image = self.to_image(fusefeature_L1)
        fusefeature_H1_image = self.to_image(fusefeature_H1)

        self.features1 = self.image_encoder(input1)
        self.features2 = self.image_encoder(input2)

        feature1_image = self.to_image(self.features1)
        feature2_image = self.to_image(self.features2)

        # self.features = self.features1 + self.features2

        fused_features = torch.cat([self.features1, self.features2, fusefeature_L1, fusefeature_H1], dim=1)
        # fused_features_image = self.to_image(fused_features)
        # 应用注意力机制进行加权
        fused_features = self.attention_fusion(fused_features)
        # 通过1x1卷积调整通道数
        self.features = self.conv_attention(fused_features)
        fused_features_image = self.to_image(self.features)

        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        #masks_rgb= onehot_to_mask(masks)
        return masks, fusefeature_L1_image, fusefeature_H1_image, feature1_image, feature2_image, fused_features_image

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = masks[0]
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G = self.criterionBCE(self.pred_mask, self.gt_mask) #(1,4,1024,1024)
        #if self.loss_mode == 'iou':
            #self.loss_G += _iou_loss(self.pred_mask, self.gt_mask)

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer.step()  # udpate G's weights

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
