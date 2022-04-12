%% ÿһ��ļ�����
%����Ȩ�غ�ƫ��
clc,clear;
% weight_bias = load('D:\WindowsFolder\Desktop\test\weight_yuan_c1.txt');

frac_bit = 3; % С��λ��
fix_prec = pow2(frac_bit); % ���㻯����

%% ��һ��
% ��Ҫ����
N_c1 = 1; %����ͨ������
M_c1 = 20;%���ͨ������
K_c1 = 5; %����˴�С
O_Size_c1 = 24;
I_Size_c1 = 28;

%% ֱ�Ӷ�ȡ
weight_c1o = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_MATLAB\param_text\weight\c1_weight.txt');  
bias_c1o = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_MATLAB\param_text\bias\c1_bias.txt');

% ���㻯����
weight_c1 = round(weight_c1o * fix_prec)/fix_prec;
bias_c1 = round(bias_c1o * fix_prec)/fix_prec;

%% ������һ������� ���ַ�����1��ԭʼͼ������ֱ��ת�� 2��������ͼ������ת
% ����ͼ��
% ����һ��ԭʼͼ������ֱ��ת
img = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_MATLAB\pix_norm.txt');

% ͼ������ 1-->10
img_index = 1;
% ��i��ͼ��
img_data = img( I_Size_c1 * I_Size_c1*(img_index-1) + 1 : I_Size_c1 * I_Size_c1 *img_index);

% ͼ�����ݴ���
img_arr = reshape(img_data, [I_Size_c1, I_Size_c1])';  % ':����ת��
img_fix_arr = round(img_arr * fix_prec) / fix_prec;
imshow(img_fix_arr);

% ǰ������
feature_c1 = conv_my(img_fix_arr, weight_c1, bias_c1, N_c1, M_c1, K_c1, O_Size_c1);
% feature_c1 = round(feature_c1 * fix_prec) / fix_prec;
feature_r1 = relu_my(feature_c1);
feature_m1 = maxpooling_my(feature_r1);


%% �ڶ���
% ��Ҫ����
N_c2 = 20; %����ͨ������
M_c2 = 50;%���ͨ������
K_c2 = 5; %����˴�С
O_Size_c2 = 8;
I_Size_c2 = 12;

% ����ԭʼȨ�ؼ�ƫ��
weight_c2o = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_MATLAB\param_text\weight\c2_weight.txt');
bias_c2o = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_MATLAB\param_text\bias\c2_bias.txt');

% ���㻯����
weight_c2 = round(weight_c2o * fix_prec)/fix_prec;
bias_c2 = round(bias_c2o * fix_prec)/fix_prec;

% ǰ������
feature_c2 = conv_my(feature_m1, weight_c2, bias_c2, N_c2, M_c2, K_c2, O_Size_c2);
% feature_c2 = round(feature_c2 * fix_prec) / fix_prec;
feature_r2 = relu_my(feature_c2);
feature_m2 = maxpooling_my(feature_r2);


%% ������
% ��Ҫ����
N_fc1 = 50; %����ͨ������
M_fc1 = 500;%���ͨ������
K_fc1 = 4; %����˴�С
O_Size_fc1 = 1;
I_Size_fc1 = 4;

% ����ԭʼȨ�ؼ�ƫ��
weight_fc1o = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_MATLAB\param_text\weight\fc1_weight.txt');
bias_fc1o = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_MATLAB\param_text\bias\fc1_bias.txt');

% ���㻯����
weight_fc1 = round(weight_fc1o * fix_prec)/fix_prec;
bias_fc1 = round(bias_fc1o * fix_prec)/fix_prec;

% ǰ������
feature_fc1 = conv_my(feature_m2, weight_fc1, bias_fc1, N_fc1, M_fc1, K_fc1, O_Size_fc1);
% feature_fc1 = round(feature_fc1 * fix_prec) / fix_prec;
feature_fc1_r1 = relu_my(feature_fc1);


%% ���Ĳ�
% ��Ҫ����
N_fc2 = 500; %����ͨ������
M_fc2 = 10;%���ͨ������
K_fc2 = 1; %����˴�С
O_Size_fc2 = 1;
I_Size_fc2 = 1;

% ����ԭʼȨ�ؼ�ƫ��
weight_fc2o = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_MATLAB\param_text\weight\fc2_weight.txt');
bias_fc2o = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_MATLAB\param_text\bias\fc2_bias.txt');

% ���㻯����
weight_fc2 = round(weight_fc2o * fix_prec)/fix_prec;
bias_fc2 = round(bias_fc2o * fix_prec)/fix_prec;

% ǰ������
feature_fc2 = conv_my(feature_fc1_r1, weight_fc2, bias_fc2, N_fc2, M_fc2, K_fc2, O_Size_fc2);
% feature_fc2 = round(feature_fc2 * fix_prec) / fix_prec;
feature_fc2_r2 = relu_my(feature_fc2);

% �������ֵ��������index��1��ʼ��
[data, pointer] = max(feature_fc2_r2)


