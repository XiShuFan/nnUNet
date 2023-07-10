"""
这个数据集用来预测validate
"""
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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xxdet.draw_box_utils import draw_objs
import json

class ValidateDataset(data.Dataset):

    def __init__(self, folder: str):
        """
        folder：图片数据所在的文件夹目录
        """

        self.folder = folder
        self.img_list = [file for file in os.listdir(folder)]

        # 统计出来的数据集中的最大边长
        self.max_h_w = (1536, 3139)

        # 从数据集中读取的图片的shape，需要进行缩放和pad了，需要重视
        self.target_shape = (640, 1280)

        # 这里写死一个图像变换
        # TODO: 直接用enumeration数据集的图像变换
        self.transforms = transforms.ComposeValidation(
            [
                # 首先是转换成tensor格式
                transforms.ToTensorValidation(),
                # 图像resize
                transforms.ResizeToShapeValidation(self.target_shape),
                # 对图像进行归一化
                transforms.UniversalNormalize(mean=[0.4427, 0.4427, 0.4427],
                                              std=[0.2383, 0.2383, 0.2383])
            ]
        )

        self.disease_classes = [
            {'name': 'Impacted'},
            {'name': 'Caries'},
            {'name': 'Periapical Lesion'},
            {'name': 'Deep Caries'}
        ]

        # 图像对应的标号
        self.img_id = {
            'val_15.png': 1,
            'val_38.png': 2,
            'val_33.png': 3,
            'val_30.png': 4,
            'val_5.png': 5,
            'val_21.png': 6,
            'val_39.png': 7,
            'val_46.png': 8,
            'val_20.png': 9,
            'val_3.png': 10,
            'val_29.png': 11,
            'val_2.png': 12,
            'val_16.png': 13,
            'val_25.png': 14,
            'val_24.png': 15,
            'val_31.png': 16,
            'val_26.png': 17,
            'val_44.png': 18,
            'val_27.png': 19,
            'val_41.png': 20,
            'val_37.png': 21,
            'val_40.png': 22,
            'val_6.png': 23,
            'val_18.png': 24,
            'val_13.png': 25,
            'val_8.png': 26,
            'val_49.png': 27,
            'val_23.png': 28,
            'val_1.png': 29,
            'val_43.png': 30,
            'val_28.png': 31,
            'val_19.png': 32,
            'val_14.png': 33,
            'val_32.png': 34,
            'val_36.png': 35,
            'val_47.png': 36,
            'val_48.png': 37,
            'val_17.png': 38,
            'val_42.png': 39,
            'val_45.png': 40,
            'val_9.png': 41,
            'val_4.png': 42,
            'val_34.png': 43,
            'val_10.png': 44,
            'val_35.png': 45,
            'val_11.png': 46,
            'val_12.png': 47,
            'val_7.png': 48,
            'val_22.png': 49,
            'val_0.png': 50
        }


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        """
        读取一张图片，进行处理之后返回图像的tensor
        """
        img = Image.open(os.path.join(self.folder, self.img_list[idx])).convert('RGB')
        w, h = img.size
        # 记录一个target信息
        target = {}
        target['origin_img_shape'] = torch.tensor(np.array([w, h]))
        target['img_name'] = self.img_list[idx]

        # 图像增强变换
        img, target = self.transforms(img, target)

        # 对于mae开源代码，直接返回图像即可
        return img, target

    def draw_pred_result(self, img, target, pred_disease, img_name, scale_factor, save_file):
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
        pred_disease = pred_disease[keep]

        # 每颗牙齿的疾病类型
        pred_disease = pred_disease.detach().cpu()
        pred_disease = np.array(pred_disease)

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
        assert len(classes) == len(pred_disease)
        combined = sorted(zip(classes, scores, boxes, pred_disease))
        filtered = {}

        for cls, score, box, disease in combined:
            if cls not in filtered or filtered[cls]['score'] < score:
                filtered[cls] = {'score': score, 'box': box, 'disease': disease}

        filtered_boxes, filtered_classes, filtered_scores, filtered_diseases = [], [], [], []
        for cls in filtered:
            filtered_classes.append(cls)
            filtered_scores.append(filtered[cls]['score'])
            filtered_boxes.append(filtered[cls]['box'])
            filtered_diseases.append(filtered[cls]['disease'])

        filtered_boxes = np.array(filtered_boxes)
        filtered_classes = np.array(filtered_classes)
        filtered_scores = np.array(filtered_scores)
        filtered_diseases = np.array(filtered_diseases)

        # 把包围盒和mask画到图片上
        pred_box = draw_objs(
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

        # 只画有疾病的牙齿包围盒，需要再次进行过滤处理
        keep = filtered_diseases != 4
        filtered_boxes = filtered_boxes[keep]
        filtered_classes = filtered_classes[keep]
        filtered_scores = filtered_scores[keep]
        filtered_diseases = filtered_diseases[keep]

        # TODO: 把结果整理成json文件，注意缩放
        result = []
        image_id = self.img_id[img_name]
        for b, c, s, d in zip(filtered_boxes, filtered_classes, filtered_scores, filtered_diseases):
            result.append(
                {
                    "image_id": int(image_id),
                    "category_id_1": int((c // 10) - 1),
                    "category_id_2": int((c % 10) - 1),
                    "category_id_3": int(d),
                    "bbox": [float(b[0] / scale_factor), float(b[1] / scale_factor),
                             float((b[2] - b[0]) / scale_factor), float((b[3] - b[1]) / scale_factor)],
                    "score": float(s)
                }
            )

        # 把疾病类型转换成文字
        filtered_diseases = [self.disease_classes[d]['name'] for d in filtered_diseases]
        pred_disease = draw_objs(
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
            draw_masks_on_image=False,
            comments=filtered_diseases
        )

        # TODO: 绘制4张图
        # make the plt figure larger
        fig = plt.figure(figsize=(24, 12))

        # 创建一个 2x2 的子图
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        ax1.imshow(img)
        ax1.set_title("origin img", fontsize=16)
        ax1.set_axis_off()

        ax2.imshow(img)
        ax2.set_title("origin img", fontsize=16)
        ax2.set_axis_off()

        ax3.imshow(pred_box)
        ax3.set_title('pred box', fontsize=16)
        ax3.set_axis_off()

        ax4.imshow(pred_disease)
        ax4.set_title('pred disease', fontsize=16)
        ax4.set_axis_off()

        plt.savefig(save_file)
        # plt.show()

        plt.close()

        return result




if __name__ == "__main__":
    dataset = ValidateDataset("D:\\xsf\\Dataset\\Oral_Panorama_Det\\validation_data\\quadrant_enumeration_disease\\xrays")

    for (img, target) in dataset:
        print(img)






