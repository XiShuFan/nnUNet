"""
这个脚本用来对validation数据进行牙齿包围盒预测
"""
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
from engine import evaluate, train_one_epoch
from models import build_model
import torch.multiprocessing as mp
import torch.distributed as dist
from xxdet.datasets.validate_dataset import ValidateDataset


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # 学习率
    parser.add_argument('--lr', default=1e-4, type=float)

    # backbone的学习率？我们这里应该不需要，直接替换成mae自监督学习好的vit就行
    parser.add_argument('--lr_backbone', default=1e-5, type=float)

    # batch size 可以设置大一点，这里设置没用
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    # epoch
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=50, type=int)
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
                        default='D:\\xsf\\Dataset\\Oral_Panorama_Det\\validation_data\\quadrant_enumeration_disease\\enumeration_val',
                        help='path where to save, empty for no saving')
    return parser


def main(args):
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 这里的class只需要设置前景即可
    model, criterion, postprocessors = build_model(args, num_classes=32, rank=0,
                                                   pretrain_mae="D:\\xsf\\Dataset\\Oral_Panorama_MAE\\save_weights\\checkpoint-220.pth")
    model.to(device)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # TODO: 加载训练好的模型
    state_dict = torch.load(
        "D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\quadrant_enumeration\\save_weights\\checkpoint0100.pth")
    model_without_ddp.load_state_dict(state_dict['model'])

    # TODO: 用我们自己的dataset
    dataset = ValidateDataset(
        "D:\\xsf\\Dataset\\Oral_Panorama_Det\\validation_data\\quadrant_enumeration_disease\\xrays")

    start_time = time.time()

    # 验证模式
    model.eval()
    criterion.eval()

    for (samples, targets) in dataset:
        samples = samples.unsqueeze(dim=0).to(device)
        targets = [{k: v.to(device) if k != 'img_name' else v for k, v in targets.items()}]

        outputs = model(samples)

        # TODO: 我们这里就直接以resize和pad之后的图像大小，之后再说怎么处理
        # [t["orig_size"] for t in targets]
        orig_target_sizes = torch.stack(
            [torch.tensor([sample.shape[1], sample.shape[2]], device=sample.device) for sample in samples], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        dataset.draw_pred_result(samples[0], results[0], os.path.join(args.output_dir, targets[0]['img_name']))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('validate time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
