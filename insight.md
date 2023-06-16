experiment_planning:

- extract_fingerprint_entry

  - 对image的非0区域提取mask，我认为没什么用。然后用mask提取整张图象的包围盒。用包围盒剪裁image和label。
  - 对剪裁后的图像提取前景体素强度值，并统计信息。（这里没有处理金属伪影，原图大小好像也没有保存）

- plan_experiment_entry、preprocess_entry

  - 设置gpu显存大小、特征图最小边长、最小的batch size、特征图最大通道数
  - fullres的重采样是使用所有图像分辨率的中位数，对于分辨率差距不是特别大的数据集是可以的
  - patch size大小初始化为 (256,256,256)
  - 只要能下采样，就下采样到不能为止，其中设置了一些经验参数。如果当前轴的==分辨率太小==或者==长度不够==，就停止下采样
  - 下采样次数可以改变，所以unet的网络架构也随着数据集改变
  - u-net使用的是 instance normalization
  - 还会估计网络消耗的显存大小，使用特征图大小来进行估计。除了GPU显存可以设置，其他很多参数都写死了（experiment_planner）。
  - 调整patch size大小尽可能和shape成比例
  - CT图像的归一化方法就是简单的标准归一化，减去均值除以标准差


- 训练过程

  - 初始化。构造model，使用SGD优化器以及LR scheduler
  - 使用dice-ce loss，并且设置了deep supervision
  - 解压数据成npy文件
  - 得到dataloader。取数据在nondet_multi_threaded_augmenter中的producer函数；data_loader_3d文件中的generat_train_batch函数；get_bbox确定剪裁区域（==随机裁剪==）
  - 随机裁剪完成之后还通过transform进行了中心裁剪得到最终的统一patch，这样的话得考虑怎么把裁剪区域的范围给提取出来呢？