%% 导入加速器中间层输出结果
conv1_output = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_matlab\graph3\conv1_output.txt');
pool1_output = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_matlab\graph3\pool1_output.txt');


%% 定点化并转二进制数
% 前向推理仿真结果
feature_r1_bin = str2double(features2bin(feature_r1));
feature_m1_bin = str2double(features2bin(feature_m1));

% 结果保存为txt
mat2txt('feature_r1_bin.txt', feature_r1_bin);
mat2txt('feature_m1_bin.txt', feature_m1_bin);

% 导入数据（一维）
feature_r1_bin_db = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_matlab\feature_r1_bin.txt');
feature_m1_bin_db = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_matlab\feature_m1_bin.txt');


%% 异或对比
xor_compare_r1 = xor(conv1_output, feature_r1_bin_db);
xor_compare_m1 = xor(pool1_output, feature_m1_bin_db);

%% 正确结果统计
r1_count = sum(xor_compare_r1==0)
m1_count = sum(xor_compare_m1==0)

r1_count_acc = sum(xor_compare_r1==0)/length(xor_compare_r1) * 100
m1_count_acc = sum(xor_compare_m1==0)/length(xor_compare_m1) * 100

% 结果保存为txt
file_1 = fopen('xor_compare_r1.txt', 'a');
fprintf(file_1, '%d\n', xor_compare_r1);
fprintf(file_1, 'r1_count = %d and r1_count_acc = %.2f', r1_count, r1_count_acc);
fclose(file_1);

file_2 = fopen('xor_compare_m1.txt', 'a');
fprintf(file_2, '%d\n', xor_compare_m1);
fprintf(file_2, 'm1_count = %d and m1_count_acc = %.2f', m1_count, m1_count_acc);
fclose(file_2);


