"""
半监督全景图分割数据集
"""
import os

import torch.utils.data as data
from PIL import Image
import transforms
from torch.utils.data.sampler import Sampler
import numpy as np
import itertools

class PanoramaDataset(data.Dataset):

    def __init__(self, pretrain: bool, img_folder, mask_folder=None):
        self.pretrain = pretrain
        self.img_folder = img_folder
        self.mask_folder = mask_folder

        # 预训练阶段需要mask
        if self.pretrain:
            assert mask_folder is not None
        else:
            assert mask_folder is None

        self.file_list = os.listdir(self.img_folder)

        # 最终需要的H和W
        self.target_shape = (640, 1280)

        # 写死一个图像变换
        self.transforms = transforms.Compose(
            [
                # 首先是转换成tensor格式
                transforms.ToTensor(),
                # 然后是进行图像的水平反转
                transforms.RandomHorizontalFlip(),
                # 再进行随机竖直反转
                transforms.RandomVerticalFlip(),
                # 对图像进行缩放到指定的大小
                transforms.ResizeToShape(self.target_shape),
                # 对图像进行归一化，这里一共是 6402张全景图的统计结果
                transforms.Normalize(mean=[0.4280888, 0.4280888, 0.42808878],
                                     std=[0.21499362, 0.21499363, 0.21499362])
            ]
        )


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, x):
        # img和mask名字一样

        if self.pretrain:
            img = Image.open(os.path.join(self.img_folder, self.file_list[x]))
            mask = Image.open(os.path.join(self.mask_folder, self.file_list[x]))

            img, mask = self.transforms(img, mask)

            # 直接通过数据增强后输出
            return img, mask

        else:
            img = Image.open(os.path.join(self.img_folder, self.file_list[x]))
            img = self.transforms(img)

            return img

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


# 验证数据集，用于查看分割效果
class ValidationDataset(data.Dataset):

    def __init__(self, img_folder):
        self.img_folder = img_folder

        self.file_list = os.listdir(self.img_folder)

        # 最终需要的H和W
        self.target_shape = (640, 1280)

        # 写死一个图像变换
        self.transforms = transforms.Compose(
            [
                # 首先是转换成tensor格式
                transforms.ToTensor(),
                # 对图像进行缩放到指定的大小
                transforms.ResizeToShape(self.target_shape),
                # 对图像进行归一化，这里一共是 6402张全景图的统计结果
                transforms.Normalize(mean=[0.4280888, 0.4280888, 0.42808878],
                                     std=[0.21499362, 0.21499363, 0.21499362])
            ]
        )


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, x):
        # img和mask名字一样
        img = Image.open(os.path.join(self.img_folder, self.file_list[x]))
        img = self.transforms(img)

        # 还需要知道当前图像的名字
        return img, self.file_list[x]

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


if __name__ == '__main__':
    pretrain_dataset = PanoramaDataset(pretrain=True, img_folder="D:\\xsf\\Dataset\\Oral_panorama_Seg\\BCP_SSL\\pretrain\\image",
                                       mask_folder="D:\\xsf\\Dataset\\Oral_panorama_Seg\\BCP_SSL\\pretrain\\mask")

    selftrain_dataset = PanoramaDataset(pretrain=False, img_folder="D:\\xsf\\Dataset\\Oral_panorama_Seg\\BCP_SSL\\selftrain")

    val_dataset = ValidationDataset("D:\\xsf\\Dataset\\Oral_panorama_Seg\\image")

    mean = [0.4280888, 0.4280888, 0.42808878]
    std = [0.21499362, 0.21499363, 0.21499362]

    img = selftrain_dataset[0]

    # 把img和mask保存成图片看一下
    img[0, :] = img[0, :] * std[0] + mean[0]
    img[1, :] = img[1, :] * std[1] + mean[1]
    img[2, :] = img[2, :] * std[2] + mean[2]
    img *= 255
    img = img.permute(1, 2, 0)
    img = img.numpy().astype(np.uint8)
    img = Image.fromarray(img)
    img.save('img.png')

    # mask = mask * 255
    # mask = mask.permute(1, 2, 0).squeeze(dim=-1)
    # mask = mask.numpy().astype(np.uint8)
    # mask = Image.fromarray(mask, mode='L')
    # mask.save('mask.png')

