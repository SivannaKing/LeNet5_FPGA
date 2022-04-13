#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@AUTHOR     WZX
@EMAIL      wuzhong_xing@126.com
@TIME&LOG   2022/4/12 - download - wzx
            -----------------
            basic function

            2022/4/13 - modify - wzx
            -----------------
            add file annotation

 TODO       google style annotation
@FUNC       convert mnist .json to .txt
@USAGE      >>> python MINIST_json_to_txt.py
            under dir [LeNet5_Pytorch]
'''


import json
import numpy as np


# 像素归一化,并保存为txt
def img_to_txt(filename, data):  # filename为写入txt文件的路径，data为要写入数据列表.
    file = open(filename, 'w')
    for i in range(len(data)):
        for j in range(len(data[i])):
            data_norm = data[i][j] / 255.0
            file.write(str(data_norm) + '\n')  # 写入文本文件
    #         file.write('the digit is:' + digit_list[i] + '\n')
    file.close()
    print("保存文件成功")


if __name__ == "__main__":
    # 读取数据集的json文件
    with open('mnistTest.json', encoding='utf-8', mode='r') as f:
        f_read = f.read()

    data = json.loads(f_read)

    # 提取指定数量的数据集标签和像素值
    img_num = 10
    digit_list = []
    pixels_list = []
    for i in range(img_num):
        digit_list += str(data['images'][i]['digit'])
        pixels_list += [data['images'][i]['pixels']]

    print(len(pixels_list))
    print(list(np.array(pixels_list)))

    img_to_txt('pix_norm.txt', pixels_list)
