"""
这个脚本统计有关牙齿类别数量
"""

import json
from pycocotools.coco import COCO
import shutil
from os.path import join
import random

# TODO: 策略是，在（50~10）之间的都扩充到50，在（10~0）之间的都扩充到10
"""
6 1  -> 10
14 1 -> 10
17 1 -> 10
18 3 -> 10
19 3 -> 10
20 6 -> 10
21 12 -> 50
22 11 -> 50
23 18 -> 50
24 17 -> 50
25 25 -> 50
26 37 -> 50
27 44 -> 50
28 74 -> 74
29 88 -> 88
30 90 -> 90
31 94 -> 94
32 107 -> 107
34 1 -> 10
40 1 -> 10
"""


def stat(ori_path: str, ori_anno: str, target_path: str, target_anno: str):
    with open(ori_anno, 'r') as f:
        info = json.load(f)

    # 准备一个target_info
    target_info = info.copy()
    target_info['images'] = []
    target_info['annotations'] = []

    # 读取出coco数据集
    coco = COCO(ori_anno)

    # 全景图的数量
    ids = list(sorted(coco.imgs.keys()))

    # 每张全景图牙齿数量统计，保存下来图像名称
    tooth_num = {}

    # 遍历每张图片，统计牙齿种类数量
    for img_id in ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        if len(ann_ids) in tooth_num:
            tooth_num[len(ann_ids)].append(img_id)
        else:
            tooth_num[len(ann_ids)] = [img_id]


    # 保存一个开始复制的图片起始编号
    img_start_id = 1
    ann_start_id = 1


    # TODO: 按照策略，（50~10）之间的扩充到50，（10~0）之间的扩充到10
    for num in tooth_num:
        img_ids = tooth_num[num]
        img_size = len(img_ids)

        # 确定随机选取的数据个数
        if img_size >= 50:
            selected_num = img_size
        elif 10 <= img_size < 50:
            selected_num = 50
        else:
            selected_num = 10

        while selected_num > 0:
            selected_img_ids = random.sample(img_ids, img_size if selected_num >= img_size else selected_num)

            selected_num -= img_size
            selected_num = max(0, selected_num)

            for id in selected_img_ids:
                img_name = coco.loadImgs(id)[0]['file_name']
                coco_img = coco.loadImgs(id)[0]
                ann_ids = coco.getAnnIds(imgIds=id)
                coco_target = coco.loadAnns(ann_ids)

                target_img_name = f'target_{img_start_id}.png'

                # TODO: 拷贝的时候需要注意一下名称
                shutil.copy(join(ori_path, img_name), join(target_path, target_img_name))

                # TODO: 把当前图片信息写入
                target_info['images'].append({
                    'file_name': target_img_name,
                    'id': img_start_id,
                    'width': coco_img['width'],
                    'height': coco_img['height']
                })

                # TODO: 拷贝完数据还不够，还需要添加一下对应的包围盒
                for t in coco_target:
                    tmp = t.copy()
                    tmp['image_id'] = img_start_id
                    tmp['id'] = ann_start_id
                    # 标注个数增加
                    ann_start_id += 1
                    target_info['annotations'].append(tmp)

                # 图片个数增加
                img_start_id += 1

    # 最后保存一下json文件
    with open(target_anno, 'w') as f:
        f.write(json.dumps(target_info, ensure_ascii=True, indent=4))



# 将处理好的图片输出到新的目标位置
if __name__ == '__main__':
    stat("D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\quadrant_enumeration\\xrays_origin",
         "D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\quadrant_enumeration\\train_quadrant_enumeration_origin.json",
         "D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\quadrant_enumeration\\xrays",
         "D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\quadrant_enumeration\\train_quadrant_enumeration.json")