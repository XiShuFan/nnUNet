"""
准备牙齿分割数据
"""
import os
import json
import random
import SimpleITK as sitk
import numpy as np
import shutil

def generate_dataset1000_tooth(origin_img_path, origin_label_path, files, imagesTr_path, imagesTs_path, labelsTr_path, labelsTs_path, json_path):
    tooth_labels = {
        11: 1,
        12: 2,
        13: 3,
        14: 4,
        15: 5,
        16: 6,
        17: 7,
        18: 8,

        21: 9,
        22: 10,
        23: 11,
        24: 12,
        25: 13,
        26: 14,
        27: 15,
        28: 16,

        31: 17,
        32: 18,
        33: 19,
        34: 20,
        35: 21,
        36: 22,
        37: 23,
        38: 24,

        41: 25,
        42: 26,
        43: 27,
        44: 28,
        45: 29,
        46: 30,
        47: 31,
        48: 32,
    }

    # 7-3比例划分训练集测试集
    json_info = {
        "channel_names": {  # formerly modalities
            "0": "CT"
        },
        "labels": {  # THIS IS DIFFERENT NOW!
            "background": 0
        },
        "numTraining": int(len(files) * 0.7),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO"
    }

    # labels 标签重写
    for label in tooth_labels:
        json_info['labels'][str(label)] = tooth_labels[label]

    for id, file in enumerate(files):
        print(id, file)
        # 读取itk格式标签数据
        origin_label = sitk.ReadImage(os.path.join(origin_label_path, file))
        origin_img = sitk.ReadImage(os.path.join(origin_img_path, file))

        # 转换成numpy格式
        origin_label_array = sitk.GetArrayFromImage(origin_label)

        # 设置范围内的标签
        target_label_array = np.zeros(origin_label_array.shape)
        for label in tooth_labels:
            target_label_array[origin_label_array == label] = tooth_labels[label]

        # 读取出itk数据
        target_label_file = sitk.GetImageFromArray(target_label_array)

        # 设置相同的属性
        target_label_file.SetSpacing(origin_img.GetSpacing())
        target_label_file.SetOrigin(origin_img.GetOrigin())
        target_label_file.SetDirection(origin_img.GetDirection())

        # 数据信息
        img_name = f'tooth_{str(id+1).zfill(4)}_0000.nii.gz'
        label_name = f'tooth_{str(id+1).zfill(4)}.nii.gz'
        if id in range(int(len(files) * 0.7)):
            # 写入文件
            shutil.copy(os.path.join(origin_img_path, file), os.path.join(imagesTr_path, img_name))
            sitk.WriteImage(target_label_file, os.path.join(labelsTr_path, label_name))
        else:
            # 写入文件
            shutil.copy(os.path.join(origin_img_path, file), os.path.join(imagesTs_path, img_name))
            sitk.WriteImage(target_label_file, os.path.join(labelsTs_path, label_name))

    with open(json_path, 'w') as f:
        json.dump(json_info, f, indent=4)


def main():
    origin_img_path = "D:\\xsf\\Dataset\\github_NC_150\\img"
    origin_label_path = "D:\\xsf\\Dataset\\github_NC_150\\label_true"

    imagesTr_path = "D:\\xsf\\Dataset\\nnUNet_raw\\Dataset1000_Tooth\\imagesTr"
    imagesTs_path = "D:\\xsf\\Dataset\\nnUNet_raw\\Dataset1000_Tooth\\imagesTs"
    labelsTr_path = "D:\\xsf\\Dataset\\nnUNet_raw\\Dataset1000_Tooth\\labelsTr"
    labelsTs_path = "D:\\xsf\\Dataset\\nnUNet_raw\\Dataset1000_Tooth\\labelsTs"
    json_path = "D:\\xsf\\Dataset\\nnUNet_raw\\Dataset1000_Tooth\\dataset.json"

    image_list = os.listdir(origin_img_path)
    label_list = os.listdir(origin_label_path)

    files = list(set(image_list).intersection(set(label_list)))
    random.shuffle(files)

    print(f'files {len(files)}')

    generate_dataset1000_tooth(origin_img_path, origin_label_path, files, imagesTr_path, imagesTs_path, labelsTr_path, labelsTs_path, json_path)


if __name__ == "__main__":
    main()




