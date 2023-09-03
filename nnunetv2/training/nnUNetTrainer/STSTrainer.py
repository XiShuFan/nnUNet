import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Union, Tuple, List

import numpy as np
import torch
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from torch._dynamo import OptimizedModule

from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import \
    LimitedLenWrapper
from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import \
    ConvertSegmentationToRegionsTransform
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
    Convert3DTo2DTransform
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from sklearn.model_selection import KFold
from torch import autocast, nn
from torch import distributed as dist
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from .nnUNetTrainer import nnUNetTrainer


class STSTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), pre_train: bool = True):
        # 直接继承父类，不需要修改
        super(STSTrainer, self).__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # 这里判断是否是预训练
        self.pre_train = pre_train

    def initialize(self):
        """
        这个函数要初始化模型。如果是pre_train，直接初始化一个老师模型；否则要同时初始化老师模型和学生模型
        """

        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)
            # 预训练，只需要构造一个老师模型
            if self.pre_train:
                # 在初始化中构造网络模型
                self.network = self.build_network_architecture(self.plans_manager, self.dataset_json,
                                                               self.configuration_manager,
                                                               self.num_input_channels,
                                                               enable_deep_supervision=True).to(self.device)
                # compile network for free speedup 编译模型加速
                if ('nnUNet_compile' in os.environ.keys()) and (
                        os.environ['nnUNet_compile'].lower() in ('true', '1', 't')):
                    self.print_to_log_file('Compiling network...')
                    self.network = torch.compile(self.network)

                self.optimizer, self.lr_scheduler = self.configure_optimizers()
                # if ddp, wrap in DDP wrapper
                if self.is_ddp:
                    self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                    self.network = DDP(self.network, device_ids=[self.local_rank])

                self.loss = self._build_loss()
                self.was_initialized = True
            # 半监督训练，需要构造老师模型和学生模型
            else:
                # 老师模型，构造完之后加载参数，并且固定参数不进行反向传播
                self.teacher = self.build_network_architecture(self.plans_manager, self.dataset_json,
                                                               self.configuration_manager,
                                                               self.num_input_channels,
                                                               enable_deep_supervision=True).to(self.device)
                # compile network for free speedup 编译模型加速
                if ('nnUNet_compile' in os.environ.keys()) and (
                        os.environ['nnUNet_compile'].lower() in ('true', '1', 't')):
                    self.print_to_log_file('Compiling network...')
                    self.teacher = torch.compile(self.teacher)

                # if ddp, wrap in DDP wrapper
                if self.is_ddp:
                    self.teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.teacher)
                    self.teacher = DDP(self.teacher, device_ids=[self.local_rank])


                # 学生模型，构造完之后加载参数，进行正常反向传播
                self.student = self.build_network_architecture(self.plans_manager, self.dataset_json,
                                                               self.configuration_manager,
                                                               self.num_input_channels,
                                                               enable_deep_supervision=True).to(self.device)
                # compile network for free speedup 编译模型加速
                if ('nnUNet_compile' in os.environ.keys()) and (
                        os.environ['nnUNet_compile'].lower() in ('true', '1', 't')):
                    self.print_to_log_file('Compiling network...')
                    self.student = torch.compile(self.student)

                # if ddp, wrap in DDP wrapper
                if self.is_ddp:
                    self.student = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.student)
                    self.student = DDP(self.student, device_ids=[self.local_rank])

                # 给学生模型用的，我认为这里是两阶段训练，所以不加载pre train记录的optimizer参数和lr_scheduler参数
                self.optimizer, self.lr_scheduler = self.configure_optimizers()
                self.loss = self._build_loss()
                self.was_initialized = True

                # 加载参数
                filename_or_checkpoint = join(self.output_folder, "checkpoint_best_pre_train.pth")
                if not isfile(filename_or_checkpoint):
                    raise RuntimeError(f'checkpoint {filename_or_checkpoint} is not a file!')
                # 加载预训练模型并且固定老师模型参数
                self.load_checkpoint_ssl(filename_or_checkpoint)

        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters() if self.pre_train else self.student.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def load_checkpoint_ssl(self, filename_or_checkpoint: str) -> None:
        """
        为ssl的老师和学生模型加载参数，并且固定老师模型
        """
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.teacher.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.teacher.module, OptimizedModule):
                self.teacher.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.teacher.module.load_state_dict(new_state_dict)
            if isinstance(self.student.module, OptimizedModule):
                self.student.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.student.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.teacher, OptimizedModule):
                self.teacher._orig_mod.load_state_dict(new_state_dict)
            else:
                self.teacher.load_state_dict(new_state_dict)
            if isinstance(self.student, OptimizedModule):
                self.student._orig_mod.load_state_dict(new_state_dict)
            else:
                self.student.load_state_dict(new_state_dict)

        # 固定住老师模型的参数
        if self.is_ddp:
            if isinstance(self.teacher.module, OptimizedModule):
                for param in self.teacher.module._orig_mod.parameters():
                    param.detach_()
            else:
                for param in self.teacher.module.parameters():
                    param.detach_()
        else:
            if isinstance(self.teacher, OptimizedModule):
                for param in self.teacher._orig_mod.parameters():
                    param.detach_()
            else:
                for param in self.teacher.parameters():
                    param.detach_()

        # 下面的参数看情况加载吧，我认为这是开启了一个全新的训练，所以不需要加载
        # self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        # if self.grad_scaler is not None:
        #     if checkpoint['grad_scaler_state'] is not None:
        #         self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

    def do_split(self):
        """
        We use all images to train, labeled images to val
        半监督，所有图片都作为训练集，有标签图像作为验证集
        :return:
        """
        case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
        labelled_keys = [case for case in case_identifiers if 'Unlabelled' not in case]
        unlabelled_keys = [case for case in case_identifiers if 'Unlabelled' in case]

        return labelled_keys, unlabelled_keys, labelled_keys

    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_labelled_keys, tr_unlabelled_keys, val_keys = self.do_split()

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr_labelled = nnUNetDataset(self.preprocessed_dataset_folder, tr_labelled_keys,
                                   folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                   num_images_properties_loading_threshold=0)

        dataset_tr_unlabelled = nnUNetDataset(self.preprocessed_dataset_folder, tr_unlabelled_keys,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                    num_images_properties_loading_threshold=0)

        dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                    num_images_properties_loading_threshold=0)
        return dataset_tr_labelled, dataset_tr_unlabelled, dataset_val

    def get_dataloaders(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            order_resampling_data=3, order_resampling_seg=1,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dl_tr_labelled, dl_tr_unlabelled, dl_val = self.get_plain_dataloaders(initial_patch_size, dim)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train_labelled = SingleThreadedAugmenter(dl_tr_labelled, tr_transforms)
            mt_gen_train_unlabelled = SingleThreadedAugmenter(dl_tr_unlabelled, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            mt_gen_train_labelled = LimitedLenWrapper(self.num_iterations_per_epoch, data_loader=dl_tr_labelled, transform=tr_transforms,
                                             num_processes=allowed_num_processes, num_cached=6, seeds=None,
                                             pin_memory=self.device.type == 'cuda', wait_time=0.02)
            mt_gen_train_unlabelled = LimitedLenWrapper(self.num_iterations_per_epoch, data_loader=dl_tr_unlabelled, transform=tr_transforms,
                                             num_processes=allowed_num_processes, num_cached=6, seeds=None,
                                             pin_memory=self.device.type == 'cuda', wait_time=0.02)
            mt_gen_val = LimitedLenWrapper(self.num_val_iterations_per_epoch, data_loader=dl_val,
                                           transform=val_transforms, num_processes=max(1, allowed_num_processes // 2),
                                           num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda',
                                           wait_time=0.02)
        return mt_gen_train_labelled, mt_gen_train_unlabelled, mt_gen_val

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr_labelled, dataset_tr_unlabelled, dataset_val = self.get_tr_and_val_datasets()

        # 获得dataloader，设置的batch size需要特别注意
        assert self.batch_size >= 4
        # 训练的时候需要同时取出有标签数据和无标签数据
        train_batch_size = self.batch_size // 2

        # 并且要注意设置为infinite，这样才能完整遍历所有的数据，因为数据不平衡
        # 有标签数据少，无标签数据多，但是我们需要把有标签数据和无标签数据一一对应读取
        # 默认就是infinite

        # 这里设置shuffle吧，只需要对训练数据
        shuffle = True

        if dim == 2:
            dl_tr_labelled = nnUNetDataLoader2D(dataset_tr_labelled, train_batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, shuffle=shuffle)

            dl_tr_unlabelled = nnUNetDataLoader2D(dataset_tr_unlabelled, train_batch_size,
                                                initial_patch_size,
                                                self.configuration_manager.patch_size,
                                                self.label_manager,
                                                oversample_foreground_percent=self.oversample_foreground_percent,
                                                sampling_probabilities=None, pad_sides=None, shuffle=shuffle)

            dl_val = nnUNetDataLoader2D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
        else:
            dl_tr_labelled = nnUNetDataLoader3D(dataset_tr_labelled, train_batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, shuffle=shuffle)

            dl_tr_unlabelled = nnUNetDataLoader3D(dataset_tr_unlabelled, train_batch_size,
                                                initial_patch_size,
                                                self.configuration_manager.patch_size,
                                                self.label_manager,
                                                oversample_foreground_percent=self.oversample_foreground_percent,
                                                sampling_probabilities=None, pad_sides=None, shuffle=shuffle)


            dl_val = nnUNetDataLoader3D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
        return dl_tr_labelled, dl_tr_unlabelled, dl_val

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.pre_train:
            if self.is_ddp:
                self.network.module.decoder.deep_supervision = enabled
            else:
                self.network.decoder.deep_supervision = enabled
        else:
            if self.is_ddp:
                self.student.module.decoder.deep_supervision = enabled
                self.teacher.module.decoder.deep_supervision = enabled
            else:
                self.student.decoder.deep_supervision = enabled
                self.teacher.decoder.deep_supervision = enabled

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()

        maybe_mkdir_p(self.output_folder)

        # make sure deep supervision is on in the network
        self.set_deep_supervision_enabled(True)

        self.print_plans()
        empty_cache(self.device)

        # maybe unpack
        if self.unpack_dataset and self.local_rank == 0:
            self.print_to_log_file('unpacking dataset...')
            unpack_dataset(self.preprocessed_dataset_folder, unpack_segmentation=True, overwrite_existing=False,
                           num_processes=max(1, round(get_allowed_n_proc_DA() // 2)))
            self.print_to_log_file('unpacking done...')

        if self.is_ddp:
            dist.barrier()

        # dataloaders must be instantiated here because they need access to the training data which may not be present
        # when doing inference

        # 在这里得到一些dataloader
        # dataloader_train_labelled和dataloader_train_unlabelled是用于加载训练数据的
        # dataloader_val是用于加载验证数据的
        self.dataloader_train_labelled, self.dataloader_train_unlabelled, self.dataloader_val = self.get_dataloaders()

        # copy plans and dataset.json so that they can be used for restoring everything we need for inference
        save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)

        # we don't really need the fingerprint but its still handy to have it with the others
        shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                    join(self.output_folder_base, 'dataset_fingerprint.json'))

        # produces a pdf in output folder
        self.plot_network_architecture()

        self._save_debug_information()

        # print(f"batch size: {self.batch_size}")
        # print(f"oversample: {self.oversample_foreground_percent}")

    def on_train_end(self):
        # 区分预训练和半监督训练
        if self.pre_train:
            self.save_checkpoint(join(self.output_folder, "checkpoint_final_pre_train.pth"))
        else:
            self.save_checkpoint(join(self.output_folder, "checkpoint_final_self_train.pth"))
        # now we can delete latest
        if self.local_rank == 0:
            if self.pre_train and isfile(join(self.output_folder, "checkpoint_latest_pre_train.pth")):
                os.remove(join(self.output_folder, "checkpoint_latest_pre_train.pth"))
            if (not self.pre_train) and isfile(join(self.output_folder, "checkpoint_latest_self_train.pth")):
                os.remove(join(self.output_folder, "checkpoint_latest_self_train.pth"))

        # shut down dataloaders
        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            if self.dataloader_train_labelled is not None:
                self.dataloader_train_labelled._finish()
            if self.dataloader_train_unlabelled is not None:
                self.dataloader_train_unlabelled._finish()
            if self.dataloader_val is not None:
                self.dataloader_val._finish()
            sys.stdout = old_stdout

        empty_cache(self.device)
        self.print_to_log_file("Training done.")

    def train_step(self, labelled_batch: dict, unlabelled_batch: dict) -> dict:
        # 有标签训练数据，确保数据没问题
        assert all([not 'Unlabelled' in file for file in labelled_batch['keys']]), "labelled数据集存在无标签数据"
        labelled_data = labelled_batch['data']
        labelled_target = labelled_batch['target']

        # 无标签训练数据，标签是无效的
        assert all(['Unlabelled' in file for file in unlabelled_batch['keys']]), "unlabelled数据集存在有标签数据"
        unlabelled_data = unlabelled_batch['data']
        unlabelled_target = unlabelled_batch['target']

        labelled_data = labelled_data.to(self.device, non_blocking=True)
        unlabelled_data = unlabelled_data.to(self.device, non_blocking=True)

        if isinstance(labelled_target, list):
            labelled_target = [i.to(self.device, non_blocking=True) for i in labelled_target]
        else:
            labelled_target = labelled_target.to(self.device, non_blocking=True)

        if isinstance(unlabelled_target, list):
            unlabelled_target = [i.to(self.device, non_blocking=True) for i in unlabelled_target]
        else:
            unlabelled_target = unlabelled_target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # 有标签数据分成两部分
            labelled_size = labelled_data.shape[0]
            assert labelled_size >= 2 and labelled_size % 2 == 0
            labelled_data_a, labelled_data_b = labelled_data[:labelled_size // 2], labelled_data[labelled_size // 2:]

            # 无标签数据分成两部分
            unlabelled_size = unlabelled_data.shape[0]
            assert unlabelled_size >= 2 and unlabelled_size % 2 == 0
            unlabelled_data_c, unlabelled_data_d = unlabelled_data[:unlabelled_size // 2], unlabelled_data[unlabelled_size // 2:]

            # 这里使用deep supervision，需要提取出来
            labelled_target_a = [labelled_a_b[:labelled_size // 2] for labelled_a_b in labelled_target]
            labelled_target_b = [labelled_a_b[labelled_size // 2:] for labelled_a_b in labelled_target]

            # 随机选取裁剪的区域mask，因为使用了deep supervision，所以需要获得多个mask
            img_masks, loss_masks = self.generate_crop_mask(labelled_target_a)

            # 这里属于pre_train的话，就只能使用有标签数据训练，小批量也没关系，不要过拟合了
            if self.pre_train:
                # 把a图像裁剪到b图像上
                crop_a_to_b_data = labelled_data_a * img_masks[0] + labelled_data_b * (1 - img_masks[0])
                crop_a_to_b_target = [target_a * loss_mask + target_b * (1 - loss_mask)
                                      for target_a, target_b, loss_mask in zip(labelled_target_a, labelled_target_b, loss_masks)]

                # 训练
                output = self.network(crop_a_to_b_data)

                # del data
                l = self.loss(output, crop_a_to_b_target)

            # 这里属于半监督训练
            else:
                # 老师模型不进行训练，所以要注意用no grad
                with torch.no_grad():
                    pre_c = self.teacher(unlabelled_data_c)
                    pre_d = self.teacher(unlabelled_data_d)
                    # 做softmax得到伪标签
                    pre_c = [torch.max(F.softmax(pred, dim=1), dim=1)[1] for pred in pre_c]
                    pre_d = [torch.max(F.softmax(pred, dim=1), dim=1)[1] for pred in pre_d]
                    # TODO: 这里我没有做连通分量检测，不知道有没有问题

                # 老师模型的任务完成了，给无标签数据打上了伪标签
                # 现在把有标签数据和无标签数据裁剪到一起，给学生模型训练

                # 无标签数据裁剪到有标签数据
                crop_c_to_a = unlabelled_data_c * img_masks[0] + labelled_data_a * (1 - img_masks[0])
                crop_c_to_a_target = [target_c * loss_mask + target_a * (1 - loss_mask)
                                      for target_c, target_a, loss_mask in zip(pre_c, labelled_target_a, loss_masks)]
                # 有标签数据裁剪到无标签数据
                crop_b_to_d = labelled_data_b * img_masks[0] + unlabelled_data_d * (1 - img_masks[0])
                crop_b_to_d_target = [target_b * loss_mask + target_d * (1 - loss_mask)
                                      for target_b, target_d, loss_mask in zip(labelled_target_b, pre_d, loss_masks)]
                # 得到输出结果
                c_to_a_output = self.student(crop_c_to_a)
                b_to_d_output = self.student(crop_b_to_d)

                # 计算loss
                l1 = self.loss(c_to_a_output, crop_c_to_a_target)
                l2 = self.loss(b_to_d_output, crop_b_to_d_target)

                l = (l1 + l2) / 2

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            if self.pre_train:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            else:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            if self.pre_train:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            else:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 12)
            self.optimizer.step()

        # 半监督训练中，每个iter，老师模型也更新
        if not self.pre_train:
            # 更新比例
            alpha = 0.99
            if self.is_ddp:
                if isinstance(self.teacher.module, OptimizedModule):
                    teacher_state_dict = self.teacher.module._orig_mod.state_dict()
                    student_state_dict = self.student.module._orig_mod.state_dict()
                    new_dict = {}
                    for key in student_state_dict:
                        new_dict[key] = alpha * teacher_state_dict[key] + (1 - alpha) * student_state_dict[key]
                    self.teacher.module._orig_mod.load_state_dict(new_dict)
                else:
                    teacher_state_dict = self.teacher.module.state_dict()
                    student_state_dict = self.student.module.state_dict()
                    new_dict = {}
                    for key in student_state_dict:
                        new_dict[key] = alpha * teacher_state_dict[key] + (1 - alpha) * student_state_dict[key]
                    self.teacher.module.load_state_dict(new_dict)
            else:
                if isinstance(self.teacher, OptimizedModule):
                    teacher_state_dict = self.teacher._orig_mod.state_dict()
                    student_state_dict = self.student._orig_mod.state_dict()
                    new_dict = {}
                    for key in student_state_dict:
                        new_dict[key] = alpha * teacher_state_dict[key] + (1 - alpha) * student_state_dict[key]
                    self.teacher._orig_mod.load_state_dict(new_dict)
                else:
                    teacher_state_dict = self.teacher.state_dict()
                    student_state_dict = self.student.state_dict()
                    new_dict = {}
                    for key in student_state_dict:
                        new_dict[key] = alpha * teacher_state_dict[key] + (1 - alpha) * student_state_dict[key]
                    self.teacher.load_state_dict(new_dict)

        return {'loss': l.detach().cpu().numpy()}

    # 对数据随机裁剪区域
    def generate_crop_mask(self, imgs):
        masks, loss_masks = [], []
        # 需要裁剪原图像的2/3大小，所以这里初始化裁剪的起始位置
        w_start = np.random.uniform(0, 1/3)
        h_start = np.random.uniform(0, 1/3)
        d_start = np.random.uniform(0, 1/3)
        for img in imgs:
            batch_size, channel, img_x, img_y, img_z = img.shape[0], img.shape[1], img.shape[2], img.shape[3], img.shape[4]
            loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
            mask = torch.ones(img_x, img_y, img_z).cuda()
            patch_x, patch_y, patch_z = int(img_x * 2 / 3), int(img_y * 2 / 3), int(img_z * 2 / 3)
            w = int(img_x * w_start)
            h = int(img_y * h_start)
            d = int(img_z * d_start)
            mask[w:w + patch_x, h:h + patch_y, d:d + patch_z] = 0
            loss_mask[:, w:w + patch_x, h:h + patch_y, d:d + patch_z] = 0
            masks.append(mask.long())
            loss_masks.append(loss_mask)
        return masks, loss_masks

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        # todo find a solution for this stupid shit
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            # 区别预训练和半监督训练的结果
            if self.pre_train:
                self.save_checkpoint(join(self.output_folder, 'checkpoint_latest_pre_train.pth'))
            else:
                self.save_checkpoint(join(self.output_folder, 'checkpoint_latest_self_train.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            if self.pre_train:
                self.save_checkpoint(join(self.output_folder, 'checkpoint_best_pre_train.pth'))
            else:
                self.save_checkpoint(join(self.output_folder, 'checkpoint_best_self_train.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    if self.pre_train:
                        mod = self.network.module
                    else:
                        mod = self.student.module
                else:
                    if self.pre_train:
                        mod = self.network
                    else:
                        mod = self.student
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def on_train_epoch_start(self):
        # 区分不同训练阶段
        if self.pre_train:
            self.network.train()
        else:
            self.teacher.train()
            self.student.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def on_validation_epoch_start(self):
        if self.pre_train:
            self.network.eval()
        else:
            self.student.eval()


    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            if self.pre_train:
                output = self.network(data)
            else:
                output = self.student(data)
            del data
            l = self.loss(output, target)

        # we only need the output with the highest output resolution
        output = output[0]
        target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, len(output.shape)))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}


    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        if self.pre_train:
            self.network.eval()
        else:
            self.student.eval()

        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_gpu=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)

        predictor.manual_initialization(self.network if self.pre_train else self.student, self.plans_manager,
                                        self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, _, val_keys = self.do_split()
            if self.is_ddp:
                val_keys = val_keys[self.local_rank:: dist.get_world_size()]

            dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                        num_images_properties_loading_threshold=0)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []

            for k in dataset_val.keys():
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                 allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                     allowed_num_queued=2)

                self.print_to_log_file(f"predicting {k}")
                data, seg, properties = dataset_val.load_case(k)

                if self.is_cascaded:
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))
                with warnings.catch_warnings():
                    # ignore 'The given NumPy array is not writable' warning
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                output_filename_truncated = join(validation_output_folder, k)

                try:
                    prediction = predictor.predict_sliding_window_return_logits(data)
                except RuntimeError:
                    predictor.perform_everything_on_gpu = False
                    prediction = predictor.predict_sliding_window_return_logits(data)
                    predictor.perform_everything_on_gpu = True

                prediction = prediction.cpu()

                # this needs to go into background processes
                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                        )
                    )
                )
                # for debug purposes
                # export_prediction(prediction_for_export, properties, self.configuration, self.plans, self.dataset_json,
                #              output_filename_truncated, save_probabilities)

                # if needed, export the softmax prediction for the next stage
                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                            next_stage_config_manager.data_identifier)

                        try:
                            # we do this so that we can use load_case and do not have to hard code how loading training cases is implemented
                            tmp = nnUNetDataset(expected_preprocessed_folder, [k],
                                                num_images_properties_loading_threshold=0)
                            d, s, p = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                f"Run the preprocessing for this configuration first!")
                            continue

                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                        output_file = join(output_folder, k + '.npz')

                        # resample_and_save(prediction, target_shape, output_file, self.plans_manager, self.configuration_manager, properties,
                        #                   self.dataset_json)
                        results.append(segmentation_export_pool.starmap_async(
                            resample_and_save, (
                                (prediction, target_shape, output_file, self.plans_manager,
                                 self.configuration_manager,
                                 properties,
                                 self.dataset_json),
                            )
                        ))

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]), also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()


    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                # 这里的逻辑是，同时取出train dataloader中有标签数据和无标签数据，用于训练
                train_outputs.append(self.train_step(next(self.dataloader_train_labelled), next(self.dataloader_train_unlabelled)))

            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()
