# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from detr_cls_engine import evaluate, train_one_epoch
from models import build_model
import torch.multiprocessing as mp
import torch.distributed as dist
from xxdet.datasets.validate_dataset import ValidateDataset
from models.detr_cls import Detr_Cls
import copy
from util.box_ops import box_cxcywh_to_xyxy


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # 学习率
    parser.add_argument('--lr', default=1e-4, type=float)

    # backbone的学习率？我们这里应该不需要，直接替换成mae自监督学习好的vit就行
    parser.add_argument('--lr_backbone', default=1e-3, type=float)

    # batch size 可以设置大一点，这里设置没用
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    # epoch
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=20, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    # 我们这里不需要
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional [TBD]backbone to use")

    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")

    # 位置编码的方式
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    # encoder层数设置，这里我们不需要
    parser.add_argument('--enc_layers', default=12, type=int,
                        help="Number of encoding layers in the transformer")

    # decoder层数设置
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")

    # 最后进行回归的MLP的维度
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")

    # 因为我们encoder用的是vit-base
    parser.add_argument('--hidden_dim', default=768, type=int,
                        help="Size of the embeddings (dimension of the transformer)")

    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")

    # 注意力机制的头个数，只需要设置给decoder
    parser.add_argument('--nheads', default=12, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    # 每张图片固定输出多少个包围盒
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation 我们这里不需要分割
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss 这里是deep supervision
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # TODO: Matcher 设置的cost权重
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # TODO: Loss coefficients 计算loss的权重
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    # TODO: 数据集我们使用自己的
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir',
                        default='D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\quadrant_enumeration_disease\\save_weights',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 这里的class只需要设置前景即可
    detector, criterion, postprocessors = build_model(args, num_classes=32, rank=0,
                                                      pretrain="D:\\xsf\\Dataset\\Oral_Panorama_MAE\\save_weights\\checkpoint-220.pth")

    # TODO: 构造一个新的网络，在检测网络之后添加一个分类头，类别数是牙齿疾病类别数
    model = Detr_Cls(detector, in_channel=3, disease_classes=4)

    # TODO: 加载训练好的模型
    state_dict = torch.load(
        "D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\quadrant_enumeration_disease\\save_weights\\checkpoint0150.pth")
    model.load_state_dict(state_dict['model'])

    model.to(device)

    # TODO: 用我们自己的dataset
    dataset = ValidateDataset(
        "D:\\xsf\\Dataset\\Oral_Panorama_Det\\validation_data\\quadrant_enumeration_disease\\xrays")

    model.eval()
    json_info = []
    for image, target in dataset:
        img_name = target['img_name']
        scale_factor = target['scale_factor'].item()
        samples = image.to(device).unsqueeze(dim=0)
        targets = [{k: v.to(device) if k != 'img_name' else v for k, v in target.items()}]

        # 得到学生模型牙齿包围盒检测结果
        outputs, image_features = model.forward_detector(samples)

        orig_target_sizes = torch.stack(
            [torch.tensor([sample.shape[1], sample.shape[2]], device=sample.device) for sample in samples], dim=0)

        # 这里使用box_roi需要把包围盒搞成xyxy的形式，并且得是0~1范围之间
        rois = [box_cxcywh_to_xyxy(tmp) for tmp in outputs['pred_boxes']]

        # 裁剪出每颗牙齿的特征图，拼接起来，预期的维度是 [B * 100, 3, 7, 7]
        box_features = [model.poller(im.unsqueeze(dim=0), [roi]) for im, roi in
                        zip(image_features, rois)]
        box_features = torch.cat(box_features, dim=0)

        # 预测得到牙齿的疾病类型
        tooth_disease_pred = model.forward_cls(box_features)

        # TODO: 我们这里就直接以resize和pad之后的图像大小，之后再说怎么处理
        # [t["orig_size"] for t in targets]
        orig_target_sizes = torch.stack(
            [torch.tensor([sample.shape[1], sample.shape[2]], device=sample.device) for sample in samples], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        # 这里需要获取预测出来的牙齿疾病类型
        tooth_disease_pred = torch.softmax(tooth_disease_pred, dim=-1)
        tooth_disease_pred = torch.argmax(tooth_disease_pred, dim=-1)

        result = dataset.draw_pred_result(samples[0], results[0], tooth_disease_pred, img_name, scale_factor,
                                          os.path.join(
                                              'D:\\xsf\\Dataset\\Oral_Panorama_Det\\validation_data\\quadrant_enumeration_disease\\disease_val',
                                              img_name))

        json_info += result

    # 最后输出到文件中
    with open('D:\\xsf\\Dataset\\Oral_Panorama_Det\\validation_data\\quadrant_enumeration_disease\\predictions.json', 'w') as f:
        f.write(json.dumps(json_info, indent=4))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
