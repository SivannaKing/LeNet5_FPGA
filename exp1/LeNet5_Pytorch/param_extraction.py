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
@FUNC       extra parameters of LeNet5
@USAGE      >>> python param_extraction.py
            under dir [LeNet5_Pytorch]
'''


import torch
import numpy as np
import pandas as pd
from LeNet5 import LeNet5
import torch.optim as optim

# 定义函数:将某一层权重/偏置参数保存到文本中 (dim=1)
def param_to_text(text_name, key_name, param):
    param_len = len(param[key_name].shape)
    with open(text_name, 'w') as f:
        if param_len == 4:
            [output_num, input_num, filter_h, filter_w] = list(param[key_name].shape)
            for i in range(output_num):
                for j in range(input_num):
                    for k in range(filter_h):
                        for v in range(filter_w):
                            f.write(str(param[key_name][i, j, k, v]) + '\n')
            f.close()

        elif param_len == 2:
            [output_num, input_num] = list(param[key_name].shape)
            for i in range(output_num):
                for j in range(input_num):
                    f.write(str(param[key_name][i, j]) + '\n')
            f.close()

        else:
            [bias_size] = list(param[key_name].shape)
            for i in range(bias_size):
                f.write(str(param[key_name][i]) + '\n')
            f.close()


if __name__ == "__main__":
    model = LeNet5()
    model.load_state_dict(torch.load('best.pt'))
    print('Successfully loaded from ckpt!\n')

    # optimal param of model
    print("Model's named_parameters:")
    for name, parameters in model.named_parameters():
        print(name, parameters)

    # Print optimizer's state_dict
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    # optimal param of model
    # save param in dict

    param = {}
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
        param[name] = parameters.detach().numpy()

    # 调用函数: weight
    param_to_text('./param_text/weight/c1_weight.txt', 'c1.c1.c1.weight', param)
    param_to_text('./param_text/weight/c2_weight.txt', 'c2.c2.c2.weight', param)
    param_to_text('./param_text/weight/fc1_weight.txt', 'fc1.fc1.fc1.weight', param)
    param_to_text('./param_text/weight/fc2_weight.txt', 'fc2.fc2.fc2.weight', param)

    # 调用函数: bias
    param_to_text('./param_text/bias/c1_bias.txt', 'c1.c1.c1.bias', param)
    param_to_text('./param_text/bias/c2_bias.txt', 'c2.c2.c2.bias', param)
    param_to_text('./param_text/bias/fc1_bias.txt', 'fc1.fc1.fc1.bias', param)
    param_to_text('./param_text/bias/fc2_bias.txt', 'fc2.fc2.fc2.bias', param)
