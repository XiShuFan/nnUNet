import numpy as np
from skimage.measure import label


def remove_small_ccl(segmentation: np.ndarray, threshold: int = None) -> np.ndarray:
    """
    把比较小的连通区域筛除
    """
    # 对所有类别进行连通分量检测，除了背景类，设置为0
    labels, ccl_num = label(label_image=segmentation, background=0, return_num=True)

    # 获得连通区域实例标签，以及出现的次数，注意把0标签排除
    classes, showup_times = np.unique(labels, return_counts=True)
    target_classes, target_showup_times = [], []

    for ccl_classes, showup in zip(classes, showup_times):
        if ccl_classes != 0:
            target_classes.append(ccl_classes)
            target_showup_times.append(showup)

    classes, showup_times = target_classes, target_showup_times

    assert ccl_num == len(classes), "连通区域数量和标签不一致"

    average = sum(showup_times) / len(showup_times)

    # 对每一个连通区域分析是否要保留
    target_seg = np.zeros(segmentation.shape, dtype=segmentation.dtype)

    # 如果没有设置threshold，计算所有连通分量的平均值，低于平均值 1/3的就要删除
    # 如果定义了threshold，那么低于threshold的连通分量需要删除
    for ccl_class, showup in zip(classes, showup_times):
        if threshold is not None:
            if showup < threshold:
                continue
        else:
            # 这里经验参数设置得不知道对不对啊
            if showup < average / 30:
                continue

        # 保存当前连通分量
        target_seg[labels == ccl_class] = segmentation[labels == ccl_class]


    return target_seg
