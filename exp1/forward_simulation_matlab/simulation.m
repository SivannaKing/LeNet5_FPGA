%% 每一层的计算结果
%加载权重和偏置
clc,clear;
% weight_bias = load('D:\WindowsFolder\Desktop\test\weight_yuan_c1.txt');

frac_bit = 3; % 小数位宽
fix_prec = pow2(frac_bit); % 定点化精度

%% 第一层
% 主要参数
N_c1 = 1; %输入通道数量
M_c1 = 20;%输出通道数量
K_c1 = 5; %卷积核大小
O_Size_c1 = 24;
I_Size_c1 = 28;

%% 直接读取
weight_c1o = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_MATLAB\param_text\weight\c1_weight.txt');  
bias_c1o = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_MATLAB\param_text\bias\c1_bias.txt');

% 定点化处理
weight_c1 = round(weight_c1o * fix_prec)/fix_prec;
bias_c1 = round(bias_c1o * fix_prec)/fix_prec;

%% 加载上一层的特征 两种方法：1、原始图像数据直接转； 2、二进制图像数据转
% 加载图像
% 方法一：原始图像数据直接转
img = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_MATLAB\pix_norm.txt');

% 图像索引 1-->10
img_index = 1;
% 第i张图像
img_data = img( I_Size_c1 * I_Size_c1*(img_index-1) + 1 : I_Size_c1 * I_Size_c1 *img_index);

% 图像数据处理
img_arr = reshape(img_data, [I_Size_c1, I_Size_c1])';  % ':矩阵转置
img_fix_arr = round(img_arr * fix_prec) / fix_prec;
imshow(img_fix_arr);

% 前向推理
feature_c1 = conv_my(img_fix_arr, weight_c1, bias_c1, N_c1, M_c1, K_c1, O_Size_c1);
% feature_c1 = round(feature_c1 * fix_prec) / fix_prec;
feature_r1 = relu_my(feature_c1);
feature_m1 = maxpooling_my(feature_r1);


%% 第二层
% 主要参数
N_c2 = 20; %输入通道数量
M_c2 = 50;%输出通道数量
K_c2 = 5; %卷积核大小
O_Size_c2 = 8;
I_Size_c2 = 12;

% 导入原始权重及偏置
weight_c2o = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_MATLAB\param_text\weight\c2_weight.txt');
bias_c2o = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_MATLAB\param_text\bias\c2_bias.txt');

% 定点化处理
weight_c2 = round(weight_c2o * fix_prec)/fix_prec;
bias_c2 = round(bias_c2o * fix_prec)/fix_prec;

% 前向推理
feature_c2 = conv_my(feature_m1, weight_c2, bias_c2, N_c2, M_c2, K_c2, O_Size_c2);
% feature_c2 = round(feature_c2 * fix_prec) / fix_prec;
feature_r2 = relu_my(feature_c2);
feature_m2 = maxpooling_my(feature_r2);


%% 第三层
% 主要参数
N_fc1 = 50; %输入通道数量
M_fc1 = 500;%输出通道数量
K_fc1 = 4; %卷积核大小
O_Size_fc1 = 1;
I_Size_fc1 = 4;

% 导入原始权重及偏置
weight_fc1o = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_MATLAB\param_text\weight\fc1_weight.txt');
bias_fc1o = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_MATLAB\param_text\bias\fc1_bias.txt');

% 定点化处理
weight_fc1 = round(weight_fc1o * fix_prec)/fix_prec;
bias_fc1 = round(bias_fc1o * fix_prec)/fix_prec;

% 前向推理
feature_fc1 = conv_my(feature_m2, weight_fc1, bias_fc1, N_fc1, M_fc1, K_fc1, O_Size_fc1);
% feature_fc1 = round(feature_fc1 * fix_prec) / fix_prec;
feature_fc1_r1 = relu_my(feature_fc1);


%% 第四层
% 主要参数
N_fc2 = 500; %输入通道数量
M_fc2 = 10;%输出通道数量
K_fc2 = 1; %卷积核大小
O_Size_fc2 = 1;
I_Size_fc2 = 1;

% 导入原始权重及偏置
weight_fc2o = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_MATLAB\param_text\weight\fc2_weight.txt');
bias_fc2o = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_MATLAB\param_text\bias\fc2_bias.txt');

% 定点化处理
weight_fc2 = round(weight_fc2o * fix_prec)/fix_prec;
bias_fc2 = round(bias_fc2o * fix_prec)/fix_prec;

% 前向推理
feature_fc2 = conv_my(feature_fc1_r1, weight_fc2, bias_fc2, N_fc2, M_fc2, K_fc2, O_Size_fc2);
% feature_fc2 = round(feature_fc2 * fix_prec) / fix_prec;
feature_fc2_r2 = relu_my(feature_fc2);

% 返回最大值及索引（index从1开始）
[data, pointer] = max(feature_fc2_r2)


