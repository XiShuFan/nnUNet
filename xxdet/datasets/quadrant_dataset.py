"""
用于象限检测的数据集
"""

import numpy
from PIL import Image
import torch.utils.data as data
import os
from xxdet.network_files import transforms
import torch.nn as nn
import random
import torch
import json
from pycocotools.coco import COCO
from xxdet.train_utils import convert_coco_poly_mask
import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from xxdet.draw_box_utils import draw_objs


class QuadrantDataset(data.Dataset):

    def __init__(self, folder: str, anno_file: str):
        """
        folder：图片数据所在的文件夹目录
        """
        super(QuadrantDataset, self).__init__()
        self.folder = folder
        self.anno_file = anno_file

        # 统计出来的数据集中的最大边长
        self.max_h_w = (1536, 3139)

        # 从数据集中读取的图片的shape
        self.target_shape = (640, 1280)

        # 这里写死一个图像变换
        self.transforms = transforms.ComposeQuadrant(
            [
                # 首先是转换成tensor格式
                transforms.ToTensorQuadrant(),
                # 然后是进行图像的水平反转，注意需要把包围盒的象限进行反转
                transforms.RandomHorizontalFlipQuadrant(),
                # 对图像进行缩放到指定的大小
                transforms.ResizeToShapeQuadrant(self.target_shape),
                # 对图像进行归一化
                transforms.UniversalNormalize(mean=[0.4427, 0.4427, 0.4427],
                                              std=[0.2383, 0.2383, 0.2383])
            ]
        )

        # 这里对json文件读取的还不完全，需要自己读取一遍
        self.coco = COCO(self.anno_file)

        with open(anno_file, 'r') as f:
            json_info = json.loads(f.read())

        # 读取象限信息
        self.quadrant_classes = json_info['categories']

        self.ids = list(sorted(self.coco.imgs.keys()))

    def parse_targets(self,
                      img_id: int,
                      coco_targets: list,
                      w: int = None,
                      h: int = None,
                      img_name: str = None):
        assert w > 0
        assert h > 0
        assert img_name is not None

        anno = [obj for obj in coco_targets]

        quadrant_boxes = np.array([np.array(obj["bbox"]) for obj in anno])

        # 统计出牙齿整体范围boxes
        boxes = np.array([[min(quadrant_boxes[:, 0]), min(quadrant_boxes[:, 1]),
                           max(quadrant_boxes[:, 0] + quadrant_boxes[:, 2]),
                           max(quadrant_boxes[:, 1] + quadrant_boxes[:, 3])]])


        boxes[:, [2, 3]] = boxes[:, [2, 3]] - boxes[:, [0, 1]]
        boxes[:, [0, 1]] += boxes[:, [2, 3]] / 2
        boxes[:, 0::2] /= w
        boxes[:, 1::2] /= h

        # TODO: 这里我们需要按照[cx, cy, w, h]处理box，并且是相对于原图的大小
        quadrant_boxes = torch.as_tensor(quadrant_boxes, dtype=torch.float32).reshape(-1, 4)
        quadrant_boxes[:, 0] += quadrant_boxes[:, 0] + quadrant_boxes[:, 2]
        quadrant_boxes[:, 1] += quadrant_boxes[:, 1] + quadrant_boxes[:, 3]
        quadrant_boxes[:, 0] /= 2
        quadrant_boxes[:, 1] /= 2
        quadrant_boxes[:, 0::2] /= w
        quadrant_boxes[:, 1::2] /= h

        # TODO: 对于DETR模型，label一定要从0开始
        quadrant_classes = [int(self.quadrant_classes[obj["category_id"]]['name']) - 1 for obj in anno]
        quadrant_classes = torch.tensor(quadrant_classes, dtype=torch.int64)

        target = {}
        target["quadrant_boxes"] = quadrant_boxes
        target["quadrant_classes"] = quadrant_classes
        target["image_id"] = torch.tensor(np.array([img_id]))

        # TODO: 观察了一下数据集，按照象限进行检测的想法不可行，因为包围盒标注太粗糙了，现在准备检测出牙齿的整体范围
        # 类别是0
        target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.tensor(np.array([0]), dtype=torch.int64)

        # 记录一下图片的原始大小，之后要进行resize，以及包围盒在原始图像上的恢复
        target['origin_img_shape'] = torch.tensor(np.array([w, h]))

        # 记录一下图像的名字，好对应起来
        target['img_name'] = img_name

        return target

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_target = coco.loadAnns(ann_ids)

        img_name = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.folder, img_name)).convert('RGB')

        w, h = img.size
        target = self.parse_targets(img_id, coco_target, w, h, img_name)

        # 数据增强
        img, target = self.transforms(img, target)

        return img, target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    @staticmethod
    # 把tensor格式的图像和包围盒展示出来，防止数据处理出现问题
    def draw_gt_box(tensor, target, save_path):
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

        # 在图像上绘制包围盒用来检查
        quadrant_boxes = target['quadrant_boxes']

        # TODO: 谁懂啊，要把相对位置转换成绝对坐标
        quadrant_boxes[:, 0::2] *= tensor.size[0]
        quadrant_boxes[:, 1::2] *= tensor.size[1]

        quadrant_boxes[:, [0, 1]] -= quadrant_boxes[:, [2, 3]] / 2
        quadrant_boxes[:, [2, 3]] += quadrant_boxes[:, [0, 1]]

        quadrant_classes = target['quadrant_classes']
        quadrant_classes = np.array(quadrant_classes)

        category_index = {str(c): c + 1 for c in quadrant_classes}

        # 把包围盒和mask画到图片上
        plot_img = draw_objs(
            image=tensor,
            boxes=quadrant_boxes,
            classes=quadrant_classes,
            scores=None,
            masks=None,
            category_index=category_index,
            line_thickness=3,
            font='arial.ttf',
            font_size=20,
            # mask没有用
            draw_masks_on_image=False
        )

        boxes = target['boxes']
        boxes[:, 0::2] *= tensor.size[0]
        boxes[:, 1::2] *= tensor.size[1]
        boxes[:, [0, 1]] -= boxes[:, [2, 3]] / 2
        boxes[:, [2, 3]] += boxes[:, [0, 1]]
        labels = np.array(target['labels'])

        # 把牙齿整体区域画到图像上
        plot_img = draw_objs(
            image=plot_img,
            boxes=boxes,
            classes=labels,
            category_index={'0': "tooth area"},
            line_thickness=3,
            font='arial.ttf',
            font_size=20,
            draw_masks_on_image=False
        )

        plt.imshow(plot_img)
        # plt.show()
        # 保存预测的图片结果
        plot_img.save(os.path.join(save_path, target['img_name']))

    @staticmethod
    def draw_pred_result(img, target, gt_boxes, gt_labels, save_file):
        img = img.detach().cpu()

        # 写入到文件中进行测试
        mean = [0.4427, 0.4427, 0.4427]
        std = [0.2383, 0.2383, 0.2383]

        img[0, :] = img[0, :] * std[0] + mean[0]
        img[1, :] = img[1, :] * std[1] + mean[1]
        img[2, :] = img[2, :] * std[2] + mean[2]

        # 别忘了乘上系数
        img *= 255

        # 将Tensor转换为PIL Image对象
        img = Image.fromarray(img.permute(1, 2, 0).numpy().astype('uint8'), 'RGB')

        # 在图像上绘制包围盒用来检查
        boxes = target['boxes'].detach().cpu()
        boxes = np.array(boxes)
        classes = target['labels'].detach().cpu()
        classes = np.array(classes)
        category_index = {str(c): c for c in classes}
        scores = target['scores'].detach().cpu()
        scores = np.array(scores)

        # 把包围盒和mask画到图片上
        plot_img = draw_objs(
            image=img,
            boxes=boxes,
            classes=classes,
            scores=scores,
            masks=None,
            category_index=category_index,
            line_thickness=3,
            font='arial.ttf',
            font_size=20,
            # mask没有用
            draw_masks_on_image=False
        )

        # 画上真值
        plot_img = draw_objs(
            image=plot_img,
            boxes=np.array(gt_boxes.detach().cpu()),
            classes=np.array([5]),
            category_index={'5': "tooth area"},
            line_thickness=3,
            font='arial.ttf',
            font_size=20,
            draw_masks_on_image=False
        )

        plt.imshow(plot_img)
        # plt.show()
        # 保存预测的图片结果
        plot_img.save(save_file)


if __name__ == "__main__":
    folder = "D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\quadrant\\xrays"
    anno_file = "D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\quadrant\\train_quadrant.json"
    dataset = QuadrantDataset(folder, anno_file)

    for (img, target) in dataset:
        dataset.draw_gt_box(img, target, 'D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\quadrant\\dataset_plot')
