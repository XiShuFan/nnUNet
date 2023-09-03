"""
直接复制三份图片
"""
import os
import shutil
from PIL import Image

def expand():
    source_path = "D:\\xsf\\Dataset\\Oral_panorama_Seg\\BCP_SSL\\pretrain\\mask"
    target_path = "D:\\xsf\\Dataset\\Oral_panorama_Seg\\BCP_SSL\\pretrain\\mask"
    # 将图片长宽放大两倍，保存到目录中
    file_list = os.listdir(source_path)
    for file in file_list:
        img = Image.open(os.path.join(source_path, file))
        w, h = img.size
        # 放大两倍
        img = img.resize((w * 2, h * 2))

        img.save(os.path.join(target_path, file))

        print(img.size)


def three_times():
    source = "D:\\xsf\\Dataset\\Oral_panorama_Seg\\BCP_SSL\\pretrain\\image"
    target = "D:\\xsf\\Dataset\\Oral_panorama_Seg\\BCP_SSL\\pretrain\\image_3x"

    file_list = os.listdir(source)

    for i in range(3):
        for file in file_list:
            shutil.copy(os.path.join(source, file), os.path.join(target, f'img{i}_{file}'))


if __name__ == '__main__':
    three_times()
