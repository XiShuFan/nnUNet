import numpy
from PIL import Image
import torch.utils.data as data
import os
from xxdet.network_files import transforms
import torch.nn as nn
import random
import torch
import torchvision
import torchvision.transforms.functional as F

class MAEDataset(data.Dataset):

    def __init__(self, folder: str):
        """
        folder：图片数据所在的文件夹目录
        """

        self.folder = folder
        self.img_list = [os.path.join(folder, file) for file in os.listdir(folder)]

        # 统计出来的数据集中的最大边长
        self.max_h_w = (1536, 3139)

        # 从数据集中读取的图片的shape，需要进行缩放和pad了，需要重视
        self.target_shape = (640, 1280)

        # patch大小是我们设定好的，不会改动
        self.patch_size = (16, 16)

        # mask的比率
        self.mask_rate = 0.25

        # 我们这里还需要生成一个随机的一维mask，长度是图片划分patch之后的个数
        self.h_patches = self.target_shape[0] // self.patch_size[0]
        self.w_patches = self.target_shape[1] // self.patch_size[1]
        self.mask_len = self.h_patches * self.w_patches

        # 这里写死一个图像变换
        self.transforms = transforms.ComposeMAE(
            [
                # 首先是转换成tensor格式
                transforms.ToTensorMAE(),
                # 然后是进行图像的水平反转，注意这里区分一下自监督学习和监督学习的两种方式
                transforms.RandomHorizontalFlipMAE(),
                # 垂直翻转
                transforms.RandomVerticalFlipMAE(),
                # 随机裁剪
                # torchvision.transforms.RandomResizedCrop(size=self.target_shape, scale=(0.2, 1.0), ratio=(1.0, 2.0),
                #                                          interpolation=F.InterpolationMode.BILINEAR, antialias=True),
                # 随机反转90度
                # torchvision.transforms.RandomApply(
                #     [torchvision.transforms.RandomRotation((90, 90))], p=0.5
                # ),
                # 图像resize
                transforms.ResizeToShapeMAE(self.target_shape),
                # 对图像进行归一化
                transforms.UniversalNormalize(mean=[0.4427, 0.4427, 0.4427],
                                              std=[0.2383, 0.2383, 0.2383])
            ]
        )


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        """
        读取一张图片，进行处理之后返回图像的tensor
        """
        img = Image.open(os.path.join(self.img_list[idx])).convert('RGB')

        # 图像增强变换
        img = self.transforms(img)

        # 保存原始图像的大小
        # origin_img_shape = img.shape

        # 最后需要把图像tensor进行pad到指定大小
        # pad = nn.ZeroPad2d((0, self.target_shape[1] - img.shape[2], 0, self.target_shape[0] - img.shape[1]))
        # img = pad(img)

        # 随机从长度中选取一定比率的token进行mask
        mask = random.sample(range(self.mask_len), int(self.mask_len * self.mask_rate))

        # 然后我这里也想直接把处理后的图片进行输出观察一下
        masked_img = img.clone()
        for pos in mask:
            h_start = pos // self.w_patches * self.patch_size[0]
            h_end = h_start + self.patch_size[0]
            w_start = pos % self.w_patches * self.patch_size[1]
            w_end = w_start + self.patch_size[1]
            masked_img[:, h_start:h_end, w_start:w_end] = 0

        # mask对应位置上应该是1才对，所以我需要重新处理一下
        tmp = numpy.zeros(self.mask_len)
        tmp[mask] = 1

        # 转化为tensor
        mask = torch.tensor(tmp)

        # 把图像名称也写上
        img_name = self.img_list[idx]

        # return img, mask, masked_img, img_name

        # 对于mae开源代码，直接返回图像即可
        return img


def tensor_to_image(tensor, img_name):
    tensor = tensor.detach().cpu()

    # 写入到文件中进行测试
    mean = [0.4427, 0.4427, 0.4427]
    std = [0.2383, 0.2383, 0.2383]

    tensor[0, :] = tensor[0, :] * std[0] + mean[0]
    tensor[1, :] = tensor[1, :] * std[1] + mean[1]
    tensor[2, :] = tensor[2, :] * std[2] + mean[2]

    # 别忘了乘上系数
    tensor *= 255

    # 将Tensor转换为PIL Image对象
    tensor = Image.fromarray(tensor.permute(1, 2, 0).numpy().astype('uint8'), 'RGB')

    # 保存为png文件，经过测试，没有大问题
    tensor.save(img_name)


if __name__ == "__main__":
    dataset = MAEDataset("D:\\xsf\Dataset\\Oral_Panorama_Det\\training_data\\quadrant\\xrays")
    img, mask, masked_img, img_name = dataset[0]

    tensor_to_image(img, "target_img.png")
    tensor_to_image(masked_img, "masked_img.png")


