"""
这个脚本用来将STS-3D数据集整理成nnUNet的格式
"""

import os
import json
import random
import SimpleITK as sitk
import numpy as np
import shutil

def generate_dataset1000_tooth(labelled_path, unlabelled_path, imagesTr_path, imagesTs_path, labelsTr_path, labelsTs_path, json_path):
    labelled_img_path = os.path.join(labelled_path, "image")
    labelled_label_path = os.path.join(labelled_path, "label")

    labelled_img_files = os.listdir(labelled_img_path)
    labelled_label_files = os.listdir(labelled_label_path)

    # 确保img和label一一对应
    assert len(list(set(labelled_img_files).intersection(set(labelled_label_files)))) == len(labelled_img_files), "img和label不是一一对应"

    unlabelled_img_files = os.listdir(unlabelled_path)

    # 训练数据是所有的有标签和无标签数据
    # 所有牙齿看成一类
    json_info = {
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "background": 0,
            "tooth": 1
        },
        "numTraining": len(labelled_img_files) + len(unlabelled_img_files),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO"
    }

    img_count = 1

    # 处理有标签数据
    for file in labelled_img_files:
        print(img_count, file)

        # 读取itk格式标签数据
        labelled_img_vtk = sitk.ReadImage(os.path.join(labelled_img_path, file))
        labelled_label_vtk = sitk.ReadImage(os.path.join(labelled_label_path, file))

        # 转换成numpy格式
        labelled_label_numpy = sitk.GetArrayFromImage(labelled_label_vtk)

        # 设置范围内的标签
        target_label_array = np.zeros(labelled_label_numpy.shape)
        target_label_array[labelled_label_numpy != 0] = 1

        # 读取出itk数据
        target_label_file = sitk.GetImageFromArray(target_label_array)

        # 设置相同的属性
        target_label_file.SetSpacing(labelled_img_vtk.GetSpacing())
        target_label_file.SetOrigin(labelled_img_vtk.GetOrigin())
        target_label_file.SetDirection(labelled_img_vtk.GetDirection())

        # 数据信息
        img_name = f'tooth_{str(img_count).zfill(4)}_0000.nii.gz'
        label_name = f'tooth_{str(img_count).zfill(4)}.nii.gz'

        # 对于有标签数据，既作为训练集，也作为验证集
        shutil.copy(os.path.join(labelled_img_path, file), os.path.join(imagesTr_path, img_name))
        sitk.WriteImage(target_label_file, os.path.join(labelsTr_path, label_name))
        shutil.copy(os.path.join(labelled_img_path, file), os.path.join(imagesTs_path, img_name))
        sitk.WriteImage(target_label_file, os.path.join(labelsTs_path, label_name))

        img_count += 1

    # 处理无标签数据
    for file in unlabelled_img_files:
        print(img_count, file)

        # 读取itk格式标签数据
        unlabelled_img_vtk = sitk.ReadImage(os.path.join(unlabelled_path, file))
        unlabelled_img_numpy = sitk.GetArrayFromImage(unlabelled_img_vtk)

        # 这是伪标签标签，没有用的，全部设置成1
        unlabelled_label_array = np.ones(unlabelled_img_numpy.shape)

        # 读取出itk数据
        target_label_file = sitk.GetImageFromArray(unlabelled_label_array)

        # 设置相同的属性
        target_label_file.SetSpacing(unlabelled_img_vtk.GetSpacing())
        target_label_file.SetOrigin(unlabelled_img_vtk.GetOrigin())
        target_label_file.SetDirection(unlabelled_img_vtk.GetDirection())

        # 数据信息
        img_name = f'toothUnlabelled_{str(img_count).zfill(4)}_0000.nii.gz'
        label_name = f'toothUnlabelled_{str(img_count).zfill(4)}.nii.gz'

        # 对于无标签数据，只作为训练集
        shutil.copy(os.path.join(unlabelled_path, file), os.path.join(imagesTr_path, img_name))
        sitk.WriteImage(target_label_file, os.path.join(labelsTr_path, label_name))

        img_count += 1

    with open(json_path, 'w') as f:
        json.dump(json_info, f, indent=4)


def main():
    # 分为标记数据和未标记数据
    labelled_path = "D:\\xsf\\Dataset\\STS-3D\\labelled"
    unlabelled_path = "D:\\xsf\\Dataset\\STS-3D\\unlabelled"

    imagesTr_path = "D:\\xsf\\Dataset\\nnUNet_raw\\Dataset1002_STS3D\\imagesTr"
    imagesTs_path = "D:\\xsf\\Dataset\\nnUNet_raw\\Dataset1002_STS3D\\imagesTs"
    labelsTr_path = "D:\\xsf\\Dataset\\nnUNet_raw\\Dataset1002_STS3D\\labelsTr"
    labelsTs_path = "D:\\xsf\\Dataset\\nnUNet_raw\\Dataset1002_STS3D\\labelsTs"
    json_path = "D:\\xsf\\Dataset\\nnUNet_raw\\Dataset1002_STS3D\\dataset.json"

    if not os.path.exists(imagesTr_path):
        os.makedirs(imagesTr_path)
    if not os.path.exists(imagesTs_path):
        os.makedirs(imagesTs_path)
    if not os.path.exists(labelsTr_path):
        os.makedirs(labelsTr_path)
    if not os.path.exists(labelsTs_path):
        os.makedirs(labelsTs_path)

    generate_dataset1000_tooth(labelled_path, unlabelled_path, imagesTr_path, imagesTs_path, labelsTr_path, labelsTs_path, json_path)


if __name__ == "__main__":
    main()
