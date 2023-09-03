- 第一步：按照nnUNet的格式要求，处理好数据。其中有标签数据作为训练集和验证集，无标签数据作为训练集（标签设置为全1），test不用于训练
- 第二步：直接运行extract_fingerprint_entry，不需要改代码，得到数据画像
- 第三步：运行plan_experiment_entry，需要改一下ExperimentPlanner中的min_batch_size，在我们的实验中，==改成4就行了==
- 第四步：运行preprocess_entry，不需要改参数
- 第五步：重写了run_training_sts.py脚本,运行run_training_entry。主要修改的地方是使用了STSTrainer

  - pre_train部分跑80个epoch
  - self_train部分跑200个epoch
- 第六步：使用后处理脚本(remove_small_ccl.py)去除比较小的连通分量。这个操作也许不一定正确