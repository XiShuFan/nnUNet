# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main_quadrant.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import torch.nn as nn

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from util.box_ops import box_cxcywh_to_xyxy


def train_one_epoch(model: torch.nn.Module, teacher: torch.nn.Module,
                    criterion, postprocessors,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('disease_ce', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = torch.stack(samples, dim=0).to(device)
        targets = [{k: v.to(device) if k != 'img_name' else v for k, v in t.items()} for t in targets]

        # TODO: 使用teacher模型得到牙齿包围盒检测结果
        teacher_outputs, _ = teacher(samples)

        # 将teacher模型的输出结果整理成确定的形式
        orig_target_sizes = torch.stack(
            [torch.tensor([sample.shape[1], sample.shape[2]], device=sample.device) for sample in samples], dim=0)
        teacher_results = postprocessors['bbox'](teacher_outputs, orig_target_sizes)

        # 将老师确定的包围盒添加到targets中，作为伪标签
        for t, teacher_o, teacher_r in zip(targets, teacher_outputs['pred_boxes'], teacher_results):
            # 构造一个概率表
            pseudo_box = {}
            for s, l, b in zip(teacher_r['scores'], teacher_r['labels'], teacher_o):
                # 置信度太低的直接丢弃
                if s.item() < 0.7:
                    continue
                # 当前牙齿标签不存在或者
                if (l.item() not in pseudo_box) or (pseudo_box[l.item()]['score'] < s.item()):
                    pseudo_box[l.item()] = {'score': s.item(), 'box': b, 'disease': 4}

            # 现在，把标准真值添加到伪标签中
            for l, b, d in zip(t['labels'], t['boxes'], t['disease_classes']):
                pseudo_box[l.item()] = {'score': 1.0, 'box': b, 'disease': d}

            # 然后用伪标签替换原始gt
            boxes = []
            labels = []
            diseases = []
            for l in pseudo_box:
                labels.append(l)
                boxes.append(pseudo_box[l]['box'])
                diseases.append(pseudo_box[l]['disease'])

            labels = torch.tensor(labels, device=device)
            boxes = torch.stack(boxes, dim=0)
            diseases = torch.tensor(diseases, device=device)

            t['labels'] = labels
            t['boxes'] = boxes
            t['disease_classes'] = diseases


        # 得到学生模型牙齿包围盒检测结果
        outputs, image_features = model(samples)

        # TODO: 计算学生输出与老师伪标签的loss，这就是半监督啊
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # TODO：接下来需要计算疾病检测分类loss
        # results = postprocessors['bbox'](outputs, orig_target_sizes)

        # 这里使用box_roi需要把包围盒搞成xyxy的形式，并且得是0~1范围之间
        # rois = [box_cxcywh_to_xyxy(tmp) for tmp in outputs['pred_boxes']]

        # 裁剪出每颗牙齿的特征图，拼接起来，预期的维度是 [B * 100, 3, 7, 7]
        # box_features = [model.poller(im.unsqueeze(dim=0), [roi]) for im, roi in zip(image_features, rois)]
        # box_features = torch.cat(box_features, dim=0)

        # 然后还需要准备每个box对应牙齿的患病类型
        # box_diseases = []
        # for t, r in zip(targets, results):
        #     # 构建牙齿类型和疾病类型查找字典
        #     tooth_disease = {i.item(): j.item() for i, j in zip(t['labels'], t['disease_classes'])}
        #     # 如果是没有患病的牙齿，类别设置为4即可
        #     box_diseases += \
        #         [tooth_disease[l.item()] if l.item() in tooth_disease else 4 for l, s in zip(r['labels'], r['scores'])]
        # box_diseases = torch.tensor(box_diseases, device=device)
        # # TODO: 我们终于可以计算loss了
        # tooth_disease_pred = model.forward_cls(box_features)
        # # 得到牙齿疾病类别损失
        # disease_ce_loss = nn.CrossEntropyLoss(reduction='none')(tooth_disease_pred, box_diseases)
        # # 把患病牙齿权重加大
        # loss_mask = torch.ones(box_diseases.shape, device=device)
        # # TODO: 这里把异常牙齿类型的比例提升了
        # loss_mask[box_diseases != 4] = 6
        # disease_ce_loss *= loss_mask
        # # 求和之后别忘了平均
        # disease_ce_loss = disease_ce_loss.sum() / len(loss_mask)
        # # TODO: 总的loss，需不需要设置超参数？
        # losses += disease_ce_loss * 10

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(disease_ce=loss_dict['loss_disease_ce'].item())

        # TODO: 当学生模型参数同步完之后，需要按照一定比例更新老师模型
        # SSL_update(model, teacher)

    # gather the stats from all processes
    # TODO: 这里同步会卡死，不知道为什么
    # metric_logger.synchronize_between_processes()
    print("stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, teacher, criterion, postprocessors, data_loader, base_ds, device, output_dir, plot_dir):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        # 基本上和训练过程一样
        samples = torch.stack(samples, dim=0).to(device)
        targets = [{k: v.to(device) if k != 'img_name' else v for k, v in t.items()} for t in targets]

        # 得到学生模型牙齿包围盒检测结果
        outputs, image_features = model(samples)

        # 老师模型的预测结果
        teacher_outputs, _ = teacher(samples)

        orig_target_sizes = torch.stack(
            [torch.tensor([sample.shape[1], sample.shape[2]], device=sample.device) for sample in samples], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        # 老师预测结果输出
        teacher_results = postprocessors['bbox'](teacher_outputs, orig_target_sizes)

        # 预测得到牙齿的疾病类型
        tooth_disease_pred = outputs['pred_disease']

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        # 包围盒真值
        gt_boxes = [t['boxes'] for t in targets]
        w, h = samples[0].shape[2], samples[0].shape[1]
        for idx in range(len(gt_boxes)):
            gt_boxes[idx][:, 0::2] *= w
            gt_boxes[idx][:, 1::2] *= h
            gt_boxes[idx][:, [0, 1]] -= gt_boxes[idx][:, [2, 3]] / 2
            gt_boxes[idx][:, [2, 3]] += gt_boxes[idx][:, [0, 1]]

        gt_labels = [t['labels'] for t in targets]

        # 这里需要获取预测出来的牙齿疾病类型
        tooth_disease_pred = torch.softmax(tooth_disease_pred, dim=-1)
        tooth_disease_pred = torch.argmax(tooth_disease_pred, dim=-1)

        # gt的疾病类型
        gt_diseases = [t['disease_classes'] for t in targets]

        data_loader.dataset.draw_pred_result(samples[0], results[0], teacher_results[0], tooth_disease_pred[0],
                                             gt_boxes[0], gt_labels[0], gt_diseases[0],
                                             os.path.join(plot_dir, targets[0]['img_name']))

        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    # TODO: 同步数据会卡死，还没找到问题
    # metric_logger.synchronize_between_processes()
    print("stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator


# 每个batch更新老师模型的参数
def SSL_update(student, teacher):
    student_state_dict = student.state_dict()
    teacher_state_dict = teacher.state_dict()
    new_dict = {}
    alpha = 0.99
    for key in student_state_dict:
        new_dict[key] = alpha * teacher_state_dict[key] + (1 - alpha) * student_state_dict[key]

    teacher.load_state_dict(new_dict)