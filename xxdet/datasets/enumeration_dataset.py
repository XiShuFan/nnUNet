"""
用于单颗牙齿检测的数据集
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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xxdet.draw_box_utils import draw_objs

class EnumerationDataset(data.Dataset):

    def __init__(self, folder: str, anno_file: str):
        """
        folder：图片数据所在的文件夹目录
        """
        super(EnumerationDataset, self).__init__()
        self.folder = folder
        self.anno_file = anno_file

        # 统计出来的数据集中的最大边长
        self.max_h_w = (1536, 3139)

        # 从数据集中读取的图片的shape
        self.target_shape = (640, 1280)

        # 这里写死一个图像变换
        self.transforms = transforms.ComposeEnumeration(
            [
                # 首先是转换成tensor格式
                transforms.ToTensorEnumeration(),
                # 然后是进行图像的水平反转，注意需要把包围盒的象限进行反转
                transforms.RandomHorizontalFlipEnumeration(),
                # 对图像进行缩放到指定的大小
                transforms.ResizeToShapeEnumeration(self.target_shape),
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
        self.quadrant_classes = json_info['categories_1']
        # 读取牙齿编号信息
        self.enumeration_classes = json_info['categories_2']

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

        boxes = [obj["bbox"] for obj in anno]

        # TODO: 这里的boxes就是每颗牙齿的包围盒，直接作为我们的回归目标就行了，不进行象限检测
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        # TODO: 这里我们需要按照[cx, cy, w, h]处理box，并且是相对于原图的大小
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 0] += boxes[:, 0] + boxes[:, 2]
        boxes[:, 1] += boxes[:, 1] + boxes[:, 3]
        boxes[:, 0] /= 2
        boxes[:, 1] /= 2
        boxes[:, 0::2] /= w
        boxes[:, 1::2] /= h

        # TODO: 这里的象限类别是FDI编号的类别
        quadrant_classes = [int(self.quadrant_classes[obj["category_id_1"]]['name']) for obj in anno]
        quadrant_classes = torch.tensor(quadrant_classes, dtype=torch.int64)

        # TODO: 这里的牙齿类别也是FDI编号的类别，我们之后进行全口牙齿类别编号
        enumeration_classes = [int(self.enumeration_classes[obj["category_id_2"]]['name']) for obj in anno]
        enumeration_classes = torch.tensor(enumeration_classes, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["quadrant_classes"] = quadrant_classes
        target["enumeration_classes"] = enumeration_classes
        target["image_id"] = torch.tensor(np.array([img_id]))

        # TODO: 每颗牙齿的标签，注意对于DETR模型来说，一定要从0开始编号
        # 一共32类，包括智齿
        target['labels'] = torch.tensor([int((qua_cls - 1) * 8 + (enu_cls - 1)) for qua_cls, enu_cls in zip(quadrant_classes, enumeration_classes)],
                                        dtype=torch.int64)

        # 保存图像原始的尺寸
        target['origin_img_shape'] = torch.tensor(np.array([w, h]))

        # 保存图像的名字
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

        # TODO: 把包围盒绘制出来
        boxes = target['boxes']
        boxes[:, 0::2] *= tensor.size[0]
        boxes[:, 1::2] *= tensor.size[1]
        boxes[:, [0, 1]] -= boxes[:, [2, 3]] / 2
        boxes[:, [2, 3]] += boxes[:, [0, 1]]

        classes = target['labels']
        classes = np.array(classes)
        # TODO: 类别还需要做一下处理，转换成FDI格式
        classes = np.array([int((cls // 8 + 1) * 10 + (cls % 8) + 1) for cls in classes])

        category_index = {str(cls): cls for cls in classes}

        # 把包围盒和mask画到图片上
        plot_img = draw_objs(
            image=tensor,
            boxes=boxes,
            classes=classes,
            scores=None,
            masks=None,
            category_index=category_index,
            line_thickness=3,
            font='arial.ttf',
            font_size=20,
            # mask没有用
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

        # TODO: 把预测概率大于0.8的以及不是空包围盒的提取出来
        keep = (target['scores'] > 0.8) & (target['labels'] != 32)
        target['scores'] = target['scores'][keep]
        target['labels'] = target['labels'][keep]
        target['boxes'] = target['boxes'][keep]

        # 在图像上绘制包围盒用来检查
        boxes = target['boxes'].detach().cpu()
        boxes = np.array(boxes)

        # TODO: 同样的这里也需要对class进行处理
        classes = target['labels'].detach().cpu()
        classes = np.array(classes)
        classes = np.array([int((cls // 8 + 1) * 10 + (cls % 8) + 1) for cls in classes])

        category_index = {str(c): c for c in classes}
        scores = target['scores'].detach().cpu()
        scores = np.array(scores)

        # TODO: 还需要过滤重复的包围盒，选择概率最大的那一个
        combined = sorted(zip(classes, scores, boxes))
        filtered = {}

        for cls, score, box in combined:
            if cls not in filtered or filtered[cls]['score'] < score:
                filtered[cls] = {'score': score, 'box': box}

        filtered_boxes, filtered_classes, filtered_scores = [], [], []
        for cls in filtered:
            filtered_classes.append(cls)
            filtered_scores.append(filtered[cls]['score'])
            filtered_boxes.append(filtered[cls]['box'])

        filtered_boxes = np.array(filtered_boxes)
        filtered_classes = np.array(filtered_classes)
        filtered_scores = np.array(filtered_scores)

        # 把包围盒和mask画到图片上
        pred_img = draw_objs(
            image=img.copy(),
            boxes=filtered_boxes,
            classes=filtered_classes,
            scores=filtered_scores,
            masks=None,
            category_index=category_index,
            line_thickness=1,
            font='arial.ttf',
            font_size=20,
            # mask没有用
            draw_masks_on_image=False
        )

        # TODO: 我这里想要画两幅图，分别展示真值和预测结果
        # 画上真值
        # gt_labels也得变
        gt_labels = gt_labels.detach().cpu()
        gt_labels = np.array(gt_labels)
        gt_labels = np.array([int((cls // 8 + 1) * 10 + (cls % 8) + 1) for cls in gt_labels])

        gt_img = draw_objs(
            image=img.copy(),
            boxes=np.array(gt_boxes.detach().cpu()),
            classes=gt_labels,
            category_index={str(cls): cls for cls in gt_labels},
            line_thickness=1,
            font='arial.ttf',
            font_size=20,
            draw_masks_on_image=False
        )

        # TODO: 并排绘制两张图像
        # make the plt figure larger
        fig = plt.figure(figsize=(12, 12))

        # 创建一个 1x2 的子图
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        ax1.imshow(gt_img)
        ax1.set_title("gt boxes", fontsize=16)
        ax1.set_axis_off()

        ax2.imshow(pred_img)
        ax2.set_title("pred boxes", fontsize=16)
        ax2.set_axis_off()

        plt.savefig(save_file)
        # plt.show()

        plt.close()


def prepare_json(anno_file: str):
    # 甚至我需要自己处理一遍json文件啊
    with open(anno_file, 'r') as f:
        info = json.load(f)

        # 添加上牙齿类别信息
        info['categories'] = [{"id": i, "name": str(i)} for i in range(0, 32)]

        # 手动添加上 category_id
        for idx in range(len(info['annotations'])):
            info['annotations'][idx]['category_id'] = info['annotations'][idx]['category_id_1'] * 8 + \
                                                      info['annotations'][idx]['category_id_2']

    with open(anno_file, 'w') as f:
        f.write(json.dumps(info, ensure_ascii=True, indent=4, separators=(',', ':')))


if __name__ == "__main__":
    folder = "D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\quadrant_enumeration\\xrays"
    anno_file = "D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\quadrant_enumeration\\train_quadrant_enumeration.json"

    dataset = EnumerationDataset(folder, anno_file)
    for (img, target) in dataset:
        dataset.draw_gt_box(img, target, "D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\quadrant_enumeration\\dataset_plot")




