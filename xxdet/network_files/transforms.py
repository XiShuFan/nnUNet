import random
from torchvision.transforms import functional as F
from typing import Tuple, List
import torch
import numpy
from torch import nn

class ComposeMAE(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class ToTensorMAE(object):
    def __call__(self, img):
        img = F.to_tensor(img)
        return img

class RandomHorizontalFlipMAE(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            image = image.flip(-1)  # 水平翻转图片
        return image


class RandomVerticalFlipMAE(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            image = image.flip(-2)  # 垂直翻转图片
        return image

class ResizeToShapeMAE(object):
    """
    将图像缩放填充到指定的大小
    """

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, image):
        scale_factor = min(self.shape[0] / image.shape[1], self.shape[1] / image.shape[2])

        image = torch.nn.functional.interpolate(
            image[None],
            size=None,
            scale_factor=scale_factor,
            mode="bilinear",
            recompute_scale_factor=False,
            align_corners=False)[0]

        # 还需要对原始图像进行插值
        pad = nn.ZeroPad2d((0, self.shape[1] - image.shape[2], 0, self.shape[0] - image.shape[1]))
        image = pad(image)

        return image


class UniversalNormalize(object):
    """
    对图片进行三通道的归一化
    """

    def __init__(self, mean: List[float], std: List[float],
                 min_clip: List[float] = None, max_clip: List[float] = None):
        self.mean = mean
        self.std = std
        self.min_clip = min_clip
        self.max_clip = max_clip

    def __call__(self, image, target=None):
        """
        主要是对image进行归一化，target不用管
        """
        # 对图像像素值进行clip裁剪
        if self.min_clip is not None and self.max_clip is not None:
            image[0, :] = torch.clamp(image[0, :], min=self.min_clip[0], max=self.max_clip[0])
            image[1, :] = torch.clamp(image[1, :], min=self.min_clip[1], max=self.max_clip[1])
            image[2, :] = torch.clamp(image[2, :], min=self.min_clip[2], max=self.max_clip[2])

        # 对图像像素值进行归一化
        image = F.normalize(image, mean=list(self.mean), std=list(self.std))

        if target is not None:
            return image, target
        else:
            return image



class ComposeQuadrant(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensorQuadrant(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


# 对象限进行调换
class RandomHorizontalFlipQuadrant(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def reverse_quadrant(self, classes: int):
        if classes == 0:
            return 1
        elif classes == 1:
            return 0
        elif classes == 2:
            return 3
        elif classes == 3:
            return 2
        else:
            raise RuntimeError("invalid quadrant")

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            bbox = target["quadrant_boxes"]
            # bbox: cx,cy,w,h 相对于原图的大小
            bbox[:, 0] = 1 - bbox[:, 0]
            target["quadrant_boxes"] = bbox

            # 反转整体的牙齿区域包围盒
            tooth_area = target['boxes']
            tooth_area[:, 0] = 1 - tooth_area[:, 0]
            target['boxes'] = tooth_area

            # 反转象限
            assert "quadrant_classes" in target
            target["quadrant_classes"] = torch.tensor([self.reverse_quadrant(classes) for classes in target["quadrant_classes"]])

        return image, target


class ResizeToShapeQuadrant(object):
    """
    将图像缩放填充到指定的大小
    """

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, image, target):
        """
        首先按照边长进行缩放，不够的地方再填充
        （1）直接按照最小的比例进行缩放
        """
        scale_factor = min(self.shape[0] / image.shape[1], self.shape[1] / image.shape[2])

        image = torch.nn.functional.interpolate(
            image[None],
            size=None,
            scale_factor=scale_factor,
            mode="bilinear",
            recompute_scale_factor=False,
            align_corners=False)[0]

        # TODO: 添加缩放的倍率
        target['scale_factor'] = torch.tensor(numpy.array([scale_factor]))

        # 别忘了对包围盒也要进行缩放，这里是相对位置缩放
        target['quadrant_boxes'][:, 1::2] *= image.shape[1] / self.shape[0]
        target['quadrant_boxes'][:, 0::2] *= image.shape[2] / self.shape[1]

        target['boxes'][:, 1::2] *= image.shape[1] / self.shape[0]
        target['boxes'][:, 0::2] *= image.shape[2] / self.shape[1]

        # 还需要对原始图像进行填充pad
        pad = nn.ZeroPad2d((0, self.shape[1] - image.shape[2], 0, self.shape[0] - image.shape[1]))
        image = pad(image)

        return image, target



class ComposeEnumeration(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensorEnumeration(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlipEnumeration(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def reverse_quadrant(self, classes: int):
        if classes == 1:
            return 2
        elif classes == 2:
            return 1
        elif classes == 3:
            return 4
        elif classes == 4:
            return 3
        else:
            raise RuntimeError("invalid quadrant")

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片

            # 水平反转包围盒
            bbox = target['boxes']
            bbox[:, 0] = 1 - bbox[:, 0]
            target['boxes'] = bbox

            # 反转象限
            assert "quadrant_classes" in target
            target["quadrant_classes"] = torch.tensor([self.reverse_quadrant(classes) for classes in target["quadrant_classes"]])

            assert 'enumeration_classes' in target

            # TODO: 反转象限之后需要重新计算一遍牙齿编号
            target['labels'] = torch.tensor([int((qua_cls - 1) * 8 + (enu_cls - 1)) for qua_cls, enu_cls in
                                             zip(target["quadrant_classes"], target['enumeration_classes'])],
                                            dtype=torch.int64)

        return image, target



class ResizeToShapeEnumeration(object):
    """
    将图像缩放填充到指定的大小
    """

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, image, target):
        """
        首先按照边长进行缩放，不够的地方再填充
        （1）直接按照最小的比例进行缩放
        """
        scale_factor = min(self.shape[0] / image.shape[1], self.shape[1] / image.shape[2])

        image = torch.nn.functional.interpolate(
            image[None],
            size=None,
            scale_factor=scale_factor,
            mode="bilinear",
            recompute_scale_factor=False,
            align_corners=False)[0]

        # TODO: 添加缩放的倍率
        target['scale_factor'] = torch.tensor(numpy.array([scale_factor]))

        # 别忘了对包围盒也要进行缩放，这里是相对位置缩放
        target['boxes'][:, 1::2] *= image.shape[1] / self.shape[0]
        target['boxes'][:, 0::2] *= image.shape[2] / self.shape[1]

        # 还需要对原始图像进行插值
        pad = nn.ZeroPad2d((0, self.shape[1] - image.shape[2], 0, self.shape[0] - image.shape[1]))
        image = pad(image)

        return image, target



class ComposeDisease(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensorDisease(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlipDisease(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def reverse_quadrant(self, classes: int):
        if classes == 1:
            return 2
        elif classes == 2:
            return 1
        elif classes == 3:
            return 4
        elif classes == 4:
            return 3
        else:
            raise RuntimeError("invalid quadrant")

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片

            # 水平反转包围盒
            bbox = target['boxes']
            bbox[:, 0] = 1 - bbox[:, 0]
            target['boxes'] = bbox

            # 反转象限
            assert "quadrant_classes" in target
            target["quadrant_classes"] = torch.tensor([self.reverse_quadrant(classes) for classes in target["quadrant_classes"]])

            assert 'enumeration_classes' in target

            # TODO: 反转象限之后需要重新计算一遍牙齿编号
            target['labels'] = torch.tensor([int((qua_cls - 1) * 8 + (enu_cls - 1)) for qua_cls, enu_cls in
                                             zip(target["quadrant_classes"], target['enumeration_classes'])],
                                            dtype=torch.int64)

        return image, target



class ResizeToShapeDisease(object):
    """
    将图像缩放填充到指定的大小
    """

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, image, target):
        """
        首先按照边长进行缩放，不够的地方再填充
        （1）直接按照最小的比例进行缩放
        """
        scale_factor = min(self.shape[0] / image.shape[1], self.shape[1] / image.shape[2])

        image = torch.nn.functional.interpolate(
            image[None],
            size=None,
            scale_factor=scale_factor,
            mode="bilinear",
            recompute_scale_factor=False,
            align_corners=False)[0]

        # TODO: 添加缩放的倍率
        target['scale_factor'] = torch.tensor(numpy.array([scale_factor]))

        # 别忘了对包围盒也要进行缩放，这里是相对位置缩放
        target['boxes'][:, 1::2] *= image.shape[1] / self.shape[0]
        target['boxes'][:, 0::2] *= image.shape[2] / self.shape[1]

        # 还需要对原始图像进行插值
        pad = nn.ZeroPad2d((0, self.shape[1] - image.shape[2], 0, self.shape[0] - image.shape[1]))
        image = pad(image)

        return image, target




class ComposeValidation(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensorValidation(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class ResizeToShapeValidation(object):
    """
    将图像缩放填充到指定的大小
    """

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, image, target):
        """
        首先按照边长进行缩放，不够的地方再填充
        （1）直接按照最小的比例进行缩放
        """
        scale_factor = min(self.shape[0] / image.shape[1], self.shape[1] / image.shape[2])

        image = torch.nn.functional.interpolate(
            image[None],
            size=None,
            scale_factor=scale_factor,
            mode="bilinear",
            recompute_scale_factor=False,
            align_corners=False)[0]

        # TODO: 添加缩放的倍率
        target['scale_factor'] = torch.tensor(numpy.array([scale_factor]))

        # 还需要对原始图像进行插值
        pad = nn.ZeroPad2d((0, self.shape[1] - image.shape[2], 0, self.shape[0] - image.shape[1]))
        image = pad(image)

        return image, target