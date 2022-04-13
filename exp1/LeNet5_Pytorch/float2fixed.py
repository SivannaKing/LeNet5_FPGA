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
@FUNC       convert float to fixed
@USAGE      >>> python float2fixed.py
            under dir [LeNet5_Pytorch]
'''


import numpy as np
import math

# 十进制转二进制
def to_bin(value, num): # 十进制数据，二进制位宽
    bin_chars = ""
    temp = value
    for i in range(num):
        bin_char = bin(temp % 2)[-1]  # python转二进制的内置函数 要求是int类型 返回的是字符串类型
        temp = temp // 2
        bin_chars = bin_char + bin_chars
    return bin_chars.upper() # 输出指定位宽的二进制字符串


# 浮点转定点方法
sign_bit = 1  # 符号位
int_bit = 4  # 整数位
frac_bit = 3  # 小数位
sum_bit = sign_bit + int_bit + frac_bit  # 二进制位宽


def float_to_fixed(float_file, fixed_file):
    float_data_file = open(float_file, 'r')
    fixed_data_file = open(fixed_file, 'a')

    float_data = float_data_file.read().strip().split('\n')
    for i in range(len(float_data)):
        data = float(float_data[i])
        x1 = data * math.pow(2, frac_bit)  # 定点化
        x2 = np.clip(x1, -math.pow(2, int_bit + frac_bit), math.pow(2, int_bit + frac_bit) - 1)  # 截断
        x3 = round(x2)  # 四舍五入取整

        #       planA:负数取原码
        # 分正负数情况讨论
        if x3 >= 0:
            output = to_bin(x3, sum_bit)
            fixed_data_file.write(output + ',' + '\n')
        else:  # 负数取原码
            x4 = x3 * (-1)
            output = to_bin(x4, sum_bit - 1)
            output = '1' + output
            fixed_data_file.write(output + ',' + '\n')

    #       planB:负数取补码
    #         output = to_bin(x3, sum_bit)
    #         fixed_data_file.write(output + '\n')
    #         fixed_data_file.write(output + ',' + '\n')

    float_data_file.close()
    fixed_data_file.close()

if __name__=="__main__":
    float_to_fixed('./param_text/weight/c1_weight.txt', 'weights_and_bias_bin.txt')
    float_to_fixed('./param_text/weight/c2_weight.txt', 'weights_and_bias_bin.txt')
    float_to_fixed('./param_text/weight/fc1_weight.txt', 'weights_and_bias_bin.txt')
    float_to_fixed('./param_text/weight/fc2_weight.txt', 'weights_and_bias_bin.txt')

    float_to_fixed('./param_text/bias/c1_bias.txt', 'weights_and_bias_bin.txt')
    float_to_fixed('./param_text/bias/c2_bias.txt', 'weights_and_bias_bin.txt')
    float_to_fixed('./param_text/bias/fc1_bias.txt', 'weights_and_bias_bin.txt')
    float_to_fixed('./param_text/bias/fc2_bias.txt', 'weights_and_bias_bin.txt')

    float_to_fixed('./pix_norm.txt', 'img_pix_bin.txt')
