"""
用于训练全景图分割
"""
from network import UNetr
import argparse
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label
from dataset import PanoramaDataset, ValidationDataset
from monai.losses import DiceCELoss
from itertools import cycle
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch.distributed as dist

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=300, help='maximum epoch number to train')
    parser.add_argument('--max_iters', type=int, default=300, help='maximum iter number to train')
    parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')

    parser.add_argument('--lr', type=float, default=0.01, help='segmentation network learning rate')

    parser.add_argument('--save_path', type=str, default='D:\\xsf\\Dataset\\Oral_panorama_Seg\\BCP_SSL\\save_weights\\pretrain')

    parser.add_argument('--pred_path', type=str, default='D:\\xsf\\Dataset\\Oral_panorama_Seg\\BCP_SSL\\save_weights\\pretrain_val')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    world_size = torch.cuda.device_count()
    # world_size = 0
    if world_size > 1:
        mp.spawn(pretrain, args=(world_size, args), nprocs=world_size, join=True)
    else:
        pretrain(0, world_size, args)


# 预训练
def pretrain(rank, world_size, args):
    device = rank

    if world_size > 1:
        torch.multiprocessing.set_start_method("spawn", force=True)
        dist.init_process_group(backend='gloo', init_method='file://D:\\xsf\\Dataset\\Oral_panorama_Seg\\BCP_SSL\\ddp.txt', world_size=world_size, rank=rank)

    # 初始化数据集
    pretrain_dataset = PanoramaDataset(pretrain=True,
                                       img_folder="D:\\xsf\\Dataset\\Oral_panorama_Seg\\BCP_SSL\\pretrain\\image",
                                       mask_folder="D:\\xsf\\Dataset\\Oral_panorama_Seg\\BCP_SSL\\pretrain\\mask")

    train_sampler = torch.utils.data.distributed.DistributedSampler(pretrain_dataset) if world_size > 1 else None

    pretrain_dataloader = torch.utils.data.DataLoader(
        dataset=pretrain_dataset,
        batch_size=4,
        shuffle=(train_sampler is None),
        pin_memory=True,
        num_workers=12,
        collate_fn=pretrain_dataset.collate_fn,
        sampler=train_sampler
    )


    # 验证数据集用于看效果
    val_dataset = ValidationDataset("D:\\xsf\\Dataset\\Oral_panorama_Seg\\image")
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if world_size > 1 else None

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=12,
                                                 sampler=val_sampler,
                                                 collate_fn=val_dataset.collate_fn)


    # 初始化模型
    unetr = UNetr()
    unetr.to(device)

    # 加载预训练参数
    # state_dict = torch.load("D:\\xsf\\Dataset\\Oral_panorama_Seg\\BCP_SSL\\save_weights\\selftrain\\epoch_10.pth")
    # state_dict = {k.split('module.')[1]: p for k, p in state_dict['state_dict'].items()}
    # unetr.load_state_dict(state_dict)

    if world_size > 1:
        # 不要忘了同步BN层（如果有的话）
        unetr = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unetr)
        unetr = torch.nn.parallel.DistributedDataParallel(unetr, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # 设置loss
    loss_fn = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, squared_pred=True, reduction='mean')

    # 设置优化器
    optimizer = torch.optim.SGD(unetr.parameters(), lr=args.lr, momentum=0.99, nesterov=True, weight_decay=1e-5)

    # 设置学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    # 记录每个iter的学习率和损失，观察收敛情况
    lr_steps, loss_steps = [], []

    # 接着训练
    resume = True
    if resume:
        state_dict = torch.load('D:\\xsf\\Dataset\\Oral_panorama_Seg\\BCP_SSL\\save_weights\\selftrain\\epoch_50.pth')
        start_epoch = state_dict['epoch']
        unetr.load_state_dict(state_dict['state_dict'])
        optimizer.load_state_dict(state_dict['optimizer'])
        lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        loss_steps = state_dict['loss_steps']
        lr_steps = state_dict['lr_steps']
    else:
        start_epoch = 0

    eval = True

    if eval:
        unetr.eval()
        with torch.no_grad():
            for image, name in val_dataloader:
                # 是一个list，直接取出来
                image = image[0]
                name = name[0]
                image = image.unsqueeze(dim=0).to(device)
                output = unetr(image)
                output = output.permute(0, 2, 3, 1)
                # 得到每个像素的类别
                # [b, H, W, 2]
                output = torch.softmax(output, dim=-1)
                output = torch.argmax(output, dim=-1)
                # 注意需要乘上255
                output = output.detach().cpu().squeeze(dim=0).numpy().astype(np.uint8) * 255
                # 保存图片
                img = Image.fromarray(output, mode='L')

                img.save(os.path.join(args.pred_path, name))
        return

    # 开始epoch
    for epoch in range(start_epoch, args.max_epochs):
        if world_size > 1:
            torch.distributed.barrier()
            train_sampler.set_epoch(epoch)
        # 训练
        unetr.train()
        for step, (images, masks) in enumerate(pretrain_dataloader):
            # 无限循环读取数据集，注意退出条件
            if step >= args.max_iters:
                break

            images = torch.stack(images, dim=0)
            masks = torch.stack(masks, dim=0)
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = unetr(images)
            iter_loss = loss_fn(outputs, masks)
            # 反向传播和优化
            optimizer.zero_grad()
            iter_loss.backward()
            optimizer.step()
            # 更新曲线
            loss_steps.append(iter_loss.item())
            lr_steps.append(lr_scheduler.get_lr())
            # 绘制lr曲线和loss曲线，这样才知道有没有收敛
            # 绘制学习率曲线
            fig, ax1 = plt.subplots()
            # 创建一个共享 x 轴的第二个 y 轴
            ax2 = ax1.twinx()
            ax1.plot(lr_steps, 'b-')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Learning rate', color='b')
            ax1.tick_params('y', colors='b')
            # 绘制损失曲线
            ax2.plot(loss_steps, 'r-')
            ax2.set_ylabel('Loss', color='r')
            ax2.tick_params('y', colors='r')
            plt.title('Learning rate and loss curve')
            plt.savefig(os.path.join(args.save_path, f'rank{rank}_lr_loss.png'))
            plt.close()
            # 打印日志
            if step % 10 == 0:
                print(f'Train epoch [{epoch}/{args.max_epochs}][{step}/{args.max_iters}]\t Lr: {lr_scheduler.get_lr()}, Loss: {iter_loss.item()}')

        # 每个epoch设置学习率调整
        lr_scheduler.step(epoch)

        # 验证并且保存模型
        if epoch % 10 == 0:
            print('start eval')
            unetr.eval()
            with torch.no_grad():
                for image, name in val_dataloader:
                    # 是一个list，直接取出来
                    image = image[0]
                    name = name[0]
                    image = image.unsqueeze(dim=0).to(device)
                    output = unetr(image)
                    output = output.permute(0, 2, 3, 1)
                    # 得到每个像素的类别
                    # [b, H, W, 2]
                    output = torch.softmax(output, dim=-1)
                    output = torch.argmax(output, dim=-1)
                    # 注意需要乘上255
                    output = output.detach().cpu().squeeze(dim=0).numpy().astype(np.uint8) * 255
                    # 保存图片
                    img = Image.fromarray(output, mode='L')

                    img.save(os.path.join(args.pred_path, name))

            # 保存当前模型
            if rank == 0:
                torch.save(
                    {
                        'epoch': epoch + 1,
                        'state_dict': unetr.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'loss_steps': loss_steps,
                        'lr_steps': lr_steps
                    }, os.path.join(args.save_path, f'epoch_{epoch}.pth')
                )
            print('end eval')




if __name__ == '__main__':
    main()