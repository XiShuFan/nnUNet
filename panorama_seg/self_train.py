"""
用于全景图分割半监督训练
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
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')

    parser.add_argument('--lr', type=float, default=0.005, help='segmentation network learning rate')

    parser.add_argument('--save_path', type=str, default='D:\\xsf\\Dataset\\Oral_panorama_Seg\\BCP_SSL\\save_weights\\selftrain')

    parser.add_argument('--pred_path', type=str, default='D:\\xsf\\Dataset\\Oral_panorama_Seg\\BCP_SSL\\save_weights\\selftrain_val')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    world_size = torch.cuda.device_count()
    # world_size = 0
    if world_size > 1:
        mp.spawn(selftrain, args=(world_size, args), nprocs=world_size, join=True)
    else:
        selftrain(0, world_size, args)


# 预训练
def selftrain(rank, world_size, args):
    device = rank

    if world_size > 1:
        torch.multiprocessing.set_start_method("spawn", force=True)
        dist.init_process_group(backend='gloo', init_method='file://D:\\xsf\\Dataset\\Oral_panorama_Seg\\BCP_SSL\\ddp.txt', world_size=world_size, rank=rank)

    # 初始化数据集
    pretrain_dataset = PanoramaDataset(pretrain=True,
                                       img_folder="D:\\xsf\\Dataset\\Oral_panorama_Seg\\BCP_SSL\\pretrain\\image_3x",
                                       mask_folder="D:\\xsf\\Dataset\\Oral_panorama_Seg\\BCP_SSL\\pretrain\\mask_3x")

    pretrain_sampler = torch.utils.data.distributed.DistributedSampler(pretrain_dataset) if world_size > 1 else None

    # 只需要每次取出两张即可
    pretrain_dataloader = torch.utils.data.DataLoader(
        dataset=pretrain_dataset,
        batch_size=4,
        shuffle=(pretrain_sampler is None),
        pin_memory=True,
        num_workers=12,
        collate_fn=pretrain_dataset.collate_fn,
        sampler=pretrain_sampler,
        prefetch_factor=2,
        drop_last=True
    )

    # 初始化半监督数据集
    selftrain_dataset = PanoramaDataset(pretrain=False,
                                        img_folder="D:\\xsf\\Dataset\\Oral_panorama_Seg\\BCP_SSL\\selftrain")

    selftrain_sampler = torch.utils.data.distributed.DistributedSampler(selftrain_dataset) if world_size > 1 else None

    selftrain_dataloader = torch.utils.data.DataLoader(
        dataset=selftrain_dataset,
        batch_size=4,
        shuffle=(selftrain_sampler is None),
        pin_memory=True,
        num_workers=12,
        sampler=selftrain_sampler,
        prefetch_factor=2,
        drop_last=True
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
                                                 collate_fn=val_dataset.collate_fn,
                                                 prefetch_factor=2)

    # 加载预训练模型
    state_dict = torch.load("D:\\xsf\\Dataset\\Oral_panorama_Seg\\BCP_SSL\\save_weights\\pretrain\\epoch_120.pth")
    state_dict = {k.split('module.')[1]: p for k, p in state_dict['state_dict'].items()}

    # 初始化学生模型
    student = UNetr()
    student.to(device)
    # 加载参数
    student.load_state_dict(state_dict)

    # 初始化老师模型
    teacher = UNetr()
    teacher.to(device)
    # 老师模型加载参数
    teacher.load_state_dict(state_dict)
    # 固定老师模型参数
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False


    if world_size > 1:
        # 不要忘了同步BN层（如果有的话）
        student = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student)
        student = torch.nn.parallel.DistributedDataParallel(student, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # 设置loss
    loss_fn = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, squared_pred=True, reduction='mean')

    # 设置优化器
    optimizer = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=0.99, nesterov=True, weight_decay=1e-5)

    # 设置学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    # 接着训练
    resume = False
    if resume:
        start_epoch = 0
    else:
        start_epoch = 0

    # 记录每个iter的学习率和损失，观察收敛情况
    lr_steps, loss_steps = [], []


    # 开始epoch
    for epoch in range(start_epoch, args.max_epochs):
        # 记得设置sampler
        if world_size > 1:
            torch.distributed.barrier()
            pretrain_sampler.set_epoch(epoch)
            selftrain_sampler.set_epoch(epoch)
        # 训练
        student.train()
        # 随机排序组合训练
        for step, (un_img, (labelled_img, labelled_mask)) in enumerate(zip(selftrain_dataloader, pretrain_dataloader)):
            # 取出图片
            patch_size = len(labelled_img)
            assert patch_size % 2 == 0

            # 图片都堆叠成patch吧
            labelled_img = torch.stack(labelled_img, dim=0)
            labelled_mask = torch.stack(labelled_mask, dim=0)

            labelled_img_a, labelled_img_b = labelled_img[0:patch_size//2], labelled_img[patch_size//2:]
            labelled_mask_a, labelled_mask_b = labelled_mask[0:patch_size//2], labelled_mask[patch_size//2:]
            labelled_img_a = labelled_img_a.to(device)
            labelled_img_b = labelled_img_b.to(device)
            labelled_mask_a = labelled_mask_a.to(device)
            labelled_mask_b = labelled_mask_b.to(device)

            un_img_c, un_img_d = un_img[0:patch_size//2], un_img[patch_size//2:]
            un_img_c = un_img_c.to(device)
            un_img_d = un_img_d.to(device)

            # 老师模型预测无标签数据mask
            un_mask_c = teacher(un_img_c)
            un_mask_c = torch.softmax(un_mask_c, dim=1)
            un_mask_c = torch.argmax(un_mask_c, dim=1, keepdim=True)

            un_mask_d = teacher(un_img_d)
            un_mask_d = torch.softmax(un_mask_d, dim=1)
            un_mask_d = torch.argmax(un_mask_d, dim=1, keepdim=True)

            # 将有标签和无标签数据进行裁剪拼接，首先需要确定随机裁剪范围，随机裁剪的范围大小是2/3
            _, channel, h, w = labelled_img_a.shape
            w_start = int(np.random.uniform(0, 1 / 3) * w)
            h_start = int(np.random.uniform(0, 1 / 3) * h)
            patch_w = int(w * 2/3)
            patch_h = int(h * 2/3)

            img_a_to_c = un_img_c
            img_a_to_c[:, :, h_start:h_start+patch_h, w_start:w_start+patch_w] = labelled_img_a[:, :, h_start:h_start+patch_h, w_start:w_start+patch_w]
            img_d_to_b = labelled_img_b
            img_d_to_b[:, :, h_start:h_start+patch_h, w_start:w_start+patch_w] = un_img_d[:, :, h_start:h_start+patch_h, w_start:w_start+patch_w]

            # 同理需要得到mask
            img_a_to_c_mask = un_mask_c
            img_a_to_c_mask[:, :, h_start:h_start+patch_h, w_start:w_start+patch_w] = labelled_mask_a[:, :, h_start:h_start+patch_h, w_start:w_start+patch_w]
            img_d_to_b_mask = labelled_mask_b
            img_d_to_b_mask[:, :, h_start:h_start+patch_h, w_start:w_start+patch_w] = un_mask_d[:, :, h_start:h_start+patch_h, w_start:w_start+patch_w]

            # 组合成patch拼接起来
            images = torch.cat([img_a_to_c, img_d_to_b], dim=0)
            masks = torch.cat([img_a_to_c_mask, img_d_to_b_mask], dim=0)
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = student(images)
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
            if step % 30 == 0:
                print(f'Train epoch [{epoch}/{args.max_epochs}][{step}]\t Lr: {lr_scheduler.get_lr()}, Loss: {iter_loss.item()}')
                # TODO: EMA更新老师模型参数
                alpha = 0.9
                student_state_dict = student.module.state_dict()
                teacher_state_dict = teacher.state_dict()
                new_state_dict = {}
                for key in student_state_dict:
                    new_state_dict[key] = alpha * teacher_state_dict[key] + (1 - alpha) * student_state_dict[key]
                teacher.load_state_dict(new_state_dict)

        # 每个epoch设置学习率调整
        lr_scheduler.step(epoch)

        # 验证并且保存模型
        if epoch % 10 == 0:
            print('start eval')
            student.eval()
            with torch.no_grad():
                for image, name in val_dataloader:
                    # 是一个list，直接取出来
                    image = image[0]
                    name = name[0]
                    image = image.unsqueeze(dim=0).to(device)
                    output = student(image)
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
                        'state_dict': student.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'loss_steps': loss_steps,
                        'lr_steps': lr_steps
                    }, os.path.join(args.save_path, f'epoch_{epoch}.pth')
                )
            print('end eval')




if __name__ == '__main__':
    main()