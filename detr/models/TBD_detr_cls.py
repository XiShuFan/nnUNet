"""
在检测网络detr之后添加一个分类头
"""
import torch
from torch import nn
import torch.nn.functional as F
from util import box_ops
import math
import torchvision

class Detr_Cls(nn.Module):

    def __init__(self, detector: nn.Module, in_channel, disease_classes):
        """
        in_channel: 输入牙齿特征图的channel
        num_classes: 牙齿疾病类型数量，不包括无疾病类型（手动添加）
        """
        super(Detr_Cls, self).__init__()

        self.detector = detector
        self.in_channel = in_channel
        self.num_classes = disease_classes + 1

        # 裁剪包围盒特征，使用roi align
        self.poller = torchvision.ops.RoIAlign(output_size=(7, 7), sampling_ratio=2, spatial_scale=1.0)

        # TODO: 构造图像分类头，获得[3,7,7]的图像特征之后，直接拉直，然后通过mlp进行分类
        self.fc1 = nn.Linear(in_features=self.in_channel * 7 * 7, out_features=256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=256, out_features=self.num_classes)


    # TODO: 这里来预测牙齿包围盒
    def forward_detector(self, x):
        # 获得检测到的牙齿包围盒以及图像的特征图，注意这里的特征图是经过了patch embed的，patch size = 16
        out, image_features = self.detector(x)
        # 返回检测到的包围盒，以及得到的图像特征（原始尺度下）
        return out, image_features

    # TODO: 这里来计算牙齿疾病类别
    def forward_cls(self, box_features):
        # 直接重排
        x = box_features.reshape(box_features.shape[0], -1)
        # 通过全连接层预测类别
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # 之后计算类别的时候还需要做softmax
        return x

