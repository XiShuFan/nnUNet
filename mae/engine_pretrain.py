# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

import matplotlib.pyplot as plt
import matplotlib

# 设置足够大的图像缓存
matplotlib.rcParams['agg.path.chunksize'] = 100000000
matplotlib.use('Agg')
plt.switch_backend('agg')

panorama_mean = torch.tensor([0.4427, 0.4427, 0.4427])
panorama_std = torch.tensor([0.2383, 0.2383, 0.2383])


def show_image(ax, image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    ax.imshow(torch.clip((image * panorama_std + panorama_mean) * 255, 0, 255).int())
    ax.set_title(title, fontsize=16)
    return


def validate_one_epoch(model: torch.nn.Module,
                       data_loader: Iterable,
                       device: torch.device,
                       epoch: int,
                       args):
    # 设置验证
    model.eval()

    for idx, samples in enumerate(data_loader):
        assert len(samples) == 1, "in validate, one image per iter"

        samples = samples.to(device, non_blocking=True)

        # run MAE
        with torch.cuda.amp.autocast():
            loss, y, mask = model(samples.float(), mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        print(f'loss is {loss_value}')

        y = model.module.unpatchify(y)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()

        # visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.module.patch_embed.patch_size[0] * model.module.patch_embed.patch_size[1] * 3)
        mask = model.module.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        x = torch.einsum('nchw->nhwc', samples).cpu()

        # masked image
        im_masked = x * (1 - mask)

        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask

        # make the plt figure larger
        fig = plt.figure(figsize=(24, 12))

        # 创建一个 2x2 的子图
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        ax1.imshow(torch.clip((x[0] * panorama_std + panorama_mean) * 255, 0, 255).int())
        ax1.set_title("original", fontsize=16)
        ax1.set_axis_off()

        ax2.imshow(torch.clip((im_masked[0] * panorama_std + panorama_mean) * 255, 0, 255).int())
        ax2.set_title("masked", fontsize=16)
        ax2.set_axis_off()

        ax3.imshow(torch.clip((y[0] * panorama_std + panorama_mean) * 255, 0, 255).int())
        ax3.set_title("reconstruction", fontsize=16)
        ax3.set_axis_off()

        ax4.imshow(torch.clip((im_paste[0] * panorama_std + panorama_mean) * 255, 0, 255).int())
        ax4.set_title("reconstruction + visible", fontsize=16)
        ax4.set_axis_off()

        plt.savefig(f"D:/xsf/Dataset/Oral_Panorama_MAE/reconstruct/device_{device.index}_img{idx}.png")
        # plt.show()

        plt.close()


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
