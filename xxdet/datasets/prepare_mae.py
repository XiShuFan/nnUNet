"""
准备MIM自监督训练数据集
"""
import shutil
import os
from PIL import Image

folders = ["D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\quadrant\\xrays",
           "D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\quadrant_enumeration\\xrays",
           "D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\quadrant_enumeration_disease\\xrays",
           "D:\\xsf\\Dataset\\Oral_Panorama_Det\\training_data\\unlabelled\\xrays",
           "D:\\xsf\\Dataset\\Oral_Panorama_Det\\validation_data\\quadrant_enumeration_disease\\xrays"]

# 为了利用nnUNet的统计能力，需要构建一个假的数据集
target_img_folder = "D:\\xsf\\Dataset\\nnUNet_raw\\Dataset1003_PanoramaDet\\imagesTr"
target_label_folder = "D:\\xsf\\Dataset\\nnUNet_raw\\Dataset1003_PanoramaDet\\labelsTr"

count = 0

for folder in folders:
    for file in os.listdir(folder):
        print(count)
        shutil.copy(os.path.join(folder, file), os.path.join(target_img_folder, f"train_{str(count).zfill(4)}_0000.png"))

        img = Image.open(os.path.join(folder, file)).convert('RGB')

        # 写入一个假的全白图片
        label = Image.new('RGB', img.size, (1, 1, 1))
        label.save(os.path.join(target_label_folder, f"train_{str(count).zfill(4)}.png"))
        count += 1


print("end")
