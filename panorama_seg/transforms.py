import random
from torchvision.transforms import functional as F
from typing import Tuple, List
import torch
import numpy
from torch import nn
import numpy as np

class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)

        if target is not None:
            return image, target
        else:
            return image


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target=None):
        image = F.to_tensor(image)
        if target is not None:
            target = F.to_tensor(target)
        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            image = image.flip(-1)  # 水平翻转图片

            if target is not None:
                target = target.flip(-1)

        return image, target


class RandomVerticalFlip(object):
    """随机竖直翻转图像"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            image = image.flip(-2)  # 竖直翻转图片

            if target is not None:
                target = target.flip(-2)

        return image, target


class ResizeToShape(object):
    """
    将图像缩放填充到指定的大小
    """

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, image, target=None):
        """
        首先按照边长进行缩放，不够的地方再填充
        （1）直接按照最小的比例进行缩放
        """
        scale_factor = min(self.shape[0] / image.shape[1], self.shape[1] / image.shape[2])

        # 注意图像使用双线性插值
        image = torch.nn.functional.interpolate(
            image[None],
            size=None,
            scale_factor=scale_factor,
            mode="bilinear",
            recompute_scale_factor=False,
            align_corners=False)[0]

        # 还需要对原始图像进行填充pad
        pad = nn.ZeroPad2d((0, self.shape[1] - image.shape[2], 0, self.shape[0] - image.shape[1]))
        image = pad(image)

        if target is not None:
            # mask使用最近邻插值
            target = torch.nn.functional.interpolate(
                target[None],
                size=None,
                scale_factor=scale_factor,
                mode="nearest",
                recompute_scale_factor=False)[0]

            target = pad(target)

            # 注意这里需要对target的标签进行处理
            target[target != 0] = 1

        return image, target


class Normalize(object):
    """
    对图片进行三通道的归一化
    """

    def __init__(self, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        # 对图像像素值进行归一化
        image = F.normalize(image, mean=list(self.mean), std=list(self.std))

        # mask不需要进行归一化
        return image, target