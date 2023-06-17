from nnunetv2.configuration import default_num_processes
from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess


def extract_fingerprint_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+', type=int, default=[1001],
                        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
                             "planning and preprocessing for these datasets. Can of course also be just one dataset")
    parser.add_argument('-fpe', type=str, required=False, default='DatasetFingerprintExtractor',
                        help='[OPTIONAL] Name of the Dataset Fingerprint Extractor class that should be used. Default is '
                             '\'DatasetFingerprintExtractor\'.')
    parser.add_argument('-np', type=int, default=default_num_processes, required=False,
                        help=f'[OPTIONAL] Number of processes used for fingerprint extraction. '
                             f'Default: {default_num_processes}')

    parser.add_argument("--verify_dataset_integrity", required=False, default=True, action="store_true",
                        help="[RECOMMENDED] set this flag to check the dataset integrity. This is useful and should be done once for "
                             "each dataset!")
    parser.add_argument("--clean", required=False, default=False, action="store_true",
                        help='[OPTIONAL] Set this flag to overwrite existing fingerprints. If this flag is not set and a '
                             'fingerprint already exists, the fingerprint extractor will not run.')
    parser.add_argument('--verbose', required=False, action='store_true',
                        help='Set this to print a lot of stuff. Useful for debugging. Will disable progrewss bar! '
                             'Recommended for cluster environments')
    args, unrecognized_args = parser.parse_known_args()

    # 我认为这里存在的问题：
    # （1）没有保存原图像大小
    # （2）没有考虑金属伪影
    extract_fingerprints(args.d, args.fpe, args.np, args.verify_dataset_integrity, args.clean, args.verbose)


def plan_experiment_entry():
    import argparse
    parser = argparse.ArgumentParser()
    # 任务编号
    parser.add_argument('-d', nargs='+', type=int, default=[1001],
                        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
                             "planning and preprocessing for these datasets. Can of course also be just one dataset")
    # 设计实验默认使用 ExperimentPlanner
    parser.add_argument('-pl', type=str, default='ExperimentPlanner', required=False,
                        help='[OPTIONAL] Name of the Experiment Planner class that should be used. Default is '
                             '\'ExperimentPlanner\'. Note: There is no longer a distinction between 2d and 3d planner. '
                             'It\'s an all in one solution now. Wuch. Such amazing.')

    # 设置gpu显存大小
    parser.add_argument('-gpu_memory_target', default=22, type=float, required=False,
                        help='[OPTIONAL] DANGER ZONE! Sets a custom GPU memory target. Default: 8 [GB]. Changing this will '
                             'affect patch and batch size and will '
                             'definitely affect your models performance! Only use this if you really know what you '
                             'are doing and NEVER use this without running the default nnU-Net first (as a baseline).')

    # 默认使用 DefaultPreprocessor
    parser.add_argument('-preprocessor_name', default='DefaultPreprocessor', type=str, required=False,
                        help='[OPTIONAL] DANGER ZONE! Sets a custom preprocessor class. This class must be located in '
                             'nnunetv2.preprocessing. Default: \'DefaultPreprocessor\'. Changing this may affect your '
                             'models performance! Only use this if you really know what you '
                             'are doing and NEVER use this without running the default nnU-Net first (as a baseline).')

    # 设置目标体素间距，默认不设置，会自动选择中位数作为重采样间距
    # 对于体素间距差距很大的数据集，需要设置一下
    parser.add_argument('-overwrite_target_spacing', default=None, nargs='+', required=False,
                        help='[OPTIONAL] DANGER ZONE! Sets a custom target spacing for the 3d_fullres and 3d_cascade_fullres '
                             'configurations. Default: None [no changes]. Changing this will affect image size and '
                             'potentially patch and batch '
                             'size. This will definitely affect your models performance! Only use this if you really '
                             'know what you are doing and NEVER use this without running the default nnU-Net first '
                             '(as a baseline). Changing the target spacing for the other configurations is currently '
                             'not implemented. New target spacing must be a list of three numbers!')

    # 如果改动了 gpu_memory_target、preprocessor_name、overwrite_target_spacing，最好给plan改一个名字，以防覆盖
    parser.add_argument('-overwrite_plans_name', default=None, required=False,
                        help='[OPTIONAL] DANGER ZONE! If you used -gpu_memory_target, -preprocessor_name or '
                             '-overwrite_target_spacing it is best practice to use -overwrite_plans_name to generate a '
                             'differently named plans file such that the nnunet default plans are not '
                             'overwritten. You will then need to specify your custom plans file with -p whenever '
                             'running other nnunet commands (training, inference etc)')

    # 设置最小的batch size，添加这个参数主要是实现STS-3D任务
    parser.add_argument('-min_batch_size', default=4, required=False, help='min_batch_size of training')

    args, unrecognized_args = parser.parse_known_args()
    plan_experiments(args.d, args.pl, args.gpu_memory_target, args.preprocessor_name, args.overwrite_target_spacing,
                     args.overwrite_plans_name, min_batch_size=args.min_batch_size)


def preprocess_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+', type=int, default=[1001],
                        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
                             "planning and preprocessing for these datasets. Can of course also be just one dataset")

    # 计划的名字，可以自己改
    parser.add_argument('-plans_name', default='nnUNetPlans', required=False,
                        help='[OPTIONAL] You can use this to specify a custom plans file that you may have generated')

    # 所有配置
    parser.add_argument('-c', required=False, default=['2d', '3d_fullres', '3d_lowres'], nargs='+',
                        help='[OPTIONAL] Configurations for which the preprocessing should be run. Default: 2d 3f_fullres '
                             '3d_lowres. 3d_cascade_fullres does not need to be specified because it uses the data '
                             'from 3f_fullres. Configurations that do not exist for some dataset will be skipped.')

    # 使用的线程数
    parser.add_argument('-np', type=int, nargs='+', default=[8, 4, 8], required=False,
                        help="[OPTIONAL] Use this to define how many processes are to be used. If this is just one number then "
                             "this number of processes is used for all configurations specified with -c. If it's a "
                             "list of numbers this list must have as many elements as there are configurations. We "
                             "then iterate over zip(configs, num_processes) to determine then umber of processes "
                             "used for each configuration. More processes is always faster (up to the number of "
                             "threads your PC can support, so 8 for a 4 core CPU with hyperthreading. If you don't "
                             "know what that is then dont touch it, or at least don't increase it!). DANGER: More "
                             "often than not the number of processes that can be used is limited by the amount of "
                             "RAM available. Image resampling takes up a lot of RAM. MONITOR RAM USAGE AND "
                             "DECREASE -np IF YOUR RAM FILLS UP TOO MUCH!. Default: 8 processes for 2d, 4 "
                             "for 3d_fullres, 8 for 3d_lowres and 4 for everything else")
    parser.add_argument('--verbose', required=False, action='store_true',
                        help='Set this to print a lot of stuff. Useful for debugging. Will disable progrewss bar! '
                             'Recommended for cluster environments')
    args, unrecognized_args = parser.parse_known_args()
    if args.np is None:
        default_np = {
            '2d': 4,
            '3d_lowres': 8,
            '3d_fullres': 4
        }
        np = {default_np[c] if c in default_np.keys() else 4 for c in args.c}
    else:
        np = args.np
    preprocess(args.d, args.plans_name, configurations=args.c, num_processes=np, verbose=args.verbose)


def plan_and_preprocess_entry():
    import argparse
    parser = argparse.ArgumentParser()

    # 需要处理的数据集，设置成自己的数据集编号就行
    parser.add_argument('-d', nargs='+', type=int, default=[1000],
                        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
                             "planning and preprocessing for these datasets. Can of course also be just one dataset")

    # 使用的finger print extractor
    parser.add_argument('-fpe', type=str, required=False, default='DatasetFingerprintExtractor',
                        help='[OPTIONAL] Name of the Dataset Fingerprint Extractor class that should be used. Default is '
                             '\'DatasetFingerprintExtractor\'.')

    # finger print extractor使用的进程个数，默认为8
    parser.add_argument('-npfp', type=int, default=8, required=False,
                        help='[OPTIONAL] Number of processes used for fingerprint extraction. Default: 8')

    # 检查数据集的一致性，很重要，最好设置
    parser.add_argument("--verify_dataset_integrity", required=False, default=True, action="store_true",
                        help="[RECOMMENDED] set this flag to check the dataset integrity. This is useful and should be done once for "
                             "each dataset!")

    # 设置这个的话，不会运行preprocess
    parser.add_argument('--no_pp', default=False, action='store_true', required=False,
                        help='[OPTIONAL] Set this to only run fingerprint extraction and experiment planning (no '
                             'preprocesing). Useful for debugging.')

    # 设置这个的话会清空finger print，更新数据集的时候需要用到
    parser.add_argument("--clean", required=False, default=False, action="store_true",
                        help='[OPTIONAL] Set this flag to overwrite existing fingerprints. If this flag is not set and a '
                             'fingerprint already exists, the fingerprint extractor will not run. REQUIRED IF YOU '
                             'CHANGE THE DATASET FINGERPRINT EXTRACTOR OR MAKE CHANGES TO THE DATASET!')

    # experiment planner类，用默认的就好
    parser.add_argument('-pl', type=str, default='ExperimentPlanner', required=False,
                        help='[OPTIONAL] Name of the Experiment Planner class that should be used. Default is '
                             '\'ExperimentPlanner\'. Note: There is no longer a distinction between 2d and 3d planner. '
                             'It\'s an all in one solution now. Wuch. Such amazing.')

    # 最好不要设置设个参数
    parser.add_argument('-gpu_memory_target', default=8, type=int, required=False,
                        help='[OPTIONAL] DANGER ZONE! Sets a custom GPU memory target. Default: 8 [GB]. Changing this will '
                             'affect patch and batch size and will '
                             'definitely affect your models performance! Only use this if you really know what you '
                             'are doing and NEVER use this without running the default nnU-Net first (as a baseline).')

    # preprocess使用的类，最好不要改
    parser.add_argument('-preprocessor_name', default='DefaultPreprocessor', type=str, required=False,
                        help='[OPTIONAL] DANGER ZONE! Sets a custom preprocessor class. This class must be located in '
                             'nnunetv2.preprocessing. Default: \'DefaultPreprocessor\'. Changing this may affect your '
                             'models performance! Only use this if you really know what you '
                             'are doing and NEVER use this without running the default nnU-Net first (as a baseline).')

    # 设置一个目标像素间距，可能会改变图像的大小，最好别做
    parser.add_argument('-overwrite_target_spacing', default=None, nargs='+', required=False,
                        help='[OPTIONAL] DANGER ZONE! Sets a custom target spacing for the 3d_fullres and 3d_cascade_fullres '
                             'configurations. Default: None [no changes]. Changing this will affect image size and '
                             'potentially patch and batch '
                             'size. This will definitely affect your models performance! Only use this if you really '
                             'know what you are doing and NEVER use this without running the default nnU-Net first '
                             '(as a baseline). Changing the target spacing for the other configurations is currently '
                             'not implemented. New target spacing must be a list of three numbers!')

    #
    parser.add_argument('-overwrite_plans_name', default='nnUNetPlans', required=False,
                        help='[OPTIONAL] uSE A CUSTOM PLANS IDENTIFIER. If you used -gpu_memory_target, '
                             '-preprocessor_name or '
                             '-overwrite_target_spacing it is best practice to use -overwrite_plans_name to generate a '
                             'differently named plans file such that the nnunet default plans are not '
                             'overwritten. You will then need to specify your custom plans file with -p whenever '
                             'running other nnunet commands (training, inference etc)')

    # 运行preprocess使用的图像类型以及分辨率，我觉得用3d_fullres就可以了
    parser.add_argument('-c', required=False, default=['2d', '3d_fullres', '3d_lowres'], nargs='+',
                        help='[OPTIONAL] Configurations for which the preprocessing should be run. Default: 2d 3f_fullres '
                             '3d_lowres. 3d_cascade_fullres does not need to be specified because it uses the data '
                             'from 3f_fullres. Configurations that do not exist for some dataset will be skipped.')
    # 不改
    parser.add_argument('-np', type=int, nargs='+', default=None, required=False,
                        help="[OPTIONAL] Use this to define how many processes are to be used. If this is just one number then "
                             "this number of processes is used for all configurations specified with -c. If it's a "
                             "list of numbers this list must have as many elements as there are configurations. We "
                             "then iterate over zip(configs, num_processes) to determine then umber of processes "
                             "used for each configuration. More processes is always faster (up to the number of "
                             "threads your PC can support, so 8 for a 4 core CPU with hyperthreading. If you don't "
                             "know what that is then dont touch it, or at least don't increase it!). DANGER: More "
                             "often than not the number of processes that can be used is limited by the amount of "
                             "RAM available. Image resampling takes up a lot of RAM. MONITOR RAM USAGE AND "
                             "DECREASE -np IF YOUR RAM FILLS UP TOO MUCH!. Default: 8 processes for 2d, 4 "
                             "for 3d_fullres, 8 for 3d_lowres and 4 for everything else")

    # 开启这个可以方便debug，但是不会显示进度条
    parser.add_argument('--verbose', required=False, action='store_true',
                        help='Set this to print a lot of stuff. Useful for debugging. Will disable progrewss bar! '
                             'Recommended for cluster environments')
    args = parser.parse_args()

    # 抽取 fingerprint
    print("Fingerprint extraction...")
    extract_fingerprints(args.d, args.fpe, args.npfp, args.verify_dataset_integrity, args.clean, args.verbose)

    # 准备实验计划
    print('Experiment planning...')
    plan_experiments(args.d, args.pl, args.gpu_memory_target, args.preprocessor_name, args.overwrite_target_spacing, args.overwrite_plans_name)

    # manage default np
    if args.np is None:
        default_np = {"2d": 8, "3d_fullres": 4, "3d_lowres": 8}
        np = [default_np[c] if c in default_np.keys() else 4 for c in args.c]
    else:
        np = args.np

    # preprocessing
    if not args.no_pp:
        print('Preprocessing...')
        preprocess(args.d, args.overwrite_plans_name, args.c, np, args.verbose)


if __name__ == '__main__':
    # plan_and_preprocess_entry()

    # 我认为这里存在的问题：
    # （1）没有保存原图像大小
    # （2）没有考虑金属伪影
    extract_fingerprint_entry()

    # 设计实验，决定体素分辨率重采样，网络架构，裁剪的patch大小等
    plan_experiment_entry()

    # 根据三个不同配置（2d, 3d_fullres, 3d_lowres），重采样数据
    preprocess_entry()
