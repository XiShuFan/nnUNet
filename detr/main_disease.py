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
from xxdet.datasets.disease_dataset import DiseaseDataset
import copy


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

    parser.add_argument('--output_dir', default='D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\quadrant_enumeration_disease\\save_weights',
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
    world_size = torch.cuda.device_count()
    # world_size = 0
    if world_size > 1:
        mp.spawn(worker, nprocs=world_size, args=(world_size, args), join=True)
    else:
        worker(0, 1, args)


def worker(rank, world_size, args):

    # TODO: 我们自己写多卡训练
    if world_size > 1:
        # 设置多卡训练后端
        torch.multiprocessing.set_start_method("spawn", force=True)
        dist.init_process_group(
            backend='gloo', init_method="file://D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\quadrant_enumeration_disease\\ddp.txt",
            world_size=world_size, rank=rank
        )

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(rank)

    # 这里的class只需要设置前景即可
    detector, criterion, postprocessors = build_model(args, num_classes=32, rank=0,
                                                   pretrain="D:\\xsf\\Dataset\\Oral_Panorama_MAE\\save_weights\\checkpoint-220.pth")

    # TODO: 加载训练好的模型
    state_dict = torch.load(
        "D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\quadrant_enumeration\\save_weights\\decoder_6_new_dataset\\checkpoint0100.pth")
    missing_keys, unexpected_keys = detector.load_state_dict(state_dict['model'], strict=False)
    assert unexpected_keys is None or len(unexpected_keys) == 0
    print(f'missing_keys: {missing_keys}')


    # TODO: 拷贝一份teacher模型，我们手动更新参数
    teacher = copy.deepcopy(detector)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.to(rank)
    teacher.eval()

    # TODO: 之前我们用roi align层提取包围盒的特征图，现在觉得这样做太蠢了，直接利用detr的架构构造额外的疾病分类头
    model = detector

    model.to(device)

    model_without_ddp = model
    if world_size > 1:
        # 不要忘了同步BN层（如果有的话）
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    # TODO: 这里的学习率调度器要不要换一个
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    # TODO: 换成余弦退火学习率，不收敛了，还是换回来
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.8)

    # TODO: 用我们自己的dataset
    train_dataset = DiseaseDataset("D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\quadrant_enumeration_disease\\xrays",
                                   "D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\quadrant_enumeration_disease\\train_quadrant_enumeration_disease.json")

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if world_size > 1 else None

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=6,
                                                   shuffle=(train_sampler is None),
                                                   pin_memory=True,
                                                   num_workers=6,
                                                   sampler=train_sampler,
                                                   collate_fn=train_dataset.collate_fn)

    # 这个我手动遍历来做部分预测
    val_dataset = DiseaseDataset("D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\quadrant_enumeration_disease\\xrays",
                                 "D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\quadrant_enumeration_disease\\train_quadrant_enumeration_disease.json")

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if world_size > 1 else None

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=6,
                                                 sampler=val_sampler,
                                                 collate_fn=train_dataset.collate_fn)

    # TODO: 这里原来要这样写
    base_ds = val_dataset.coco

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if world_size > 1:
            torch.distributed.barrier()
            train_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model_without_ddp, teacher, criterion, postprocessors, train_dataloader, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()

        # 验证一次
        if epoch % 10 == 0:
            test_stats, coco_evaluator = evaluate(
                model_without_ddp, teacher, criterion, postprocessors, val_dataloader, base_ds, device, args.output_dir,
                plot_dir="D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\quadrant_enumeration_disease\\val"
            )

            # 在GPU0上保存权重
            if rank == 0:
                checkpoint_path = os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth')
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                             'n_parameters': n_parameters}

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    Path(os.path.join(args.output_dir, 'eval')).mkdir(parents=True, exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                       os.path.join(args.output_dir, "eval", name))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
