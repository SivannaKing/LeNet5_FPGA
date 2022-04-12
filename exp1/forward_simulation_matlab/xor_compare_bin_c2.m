%% 导入加速器中间层输出结果
conv2_output = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_matlab\graph3\conv2_output.txt');
pool2_output = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_matlab\graph3\pool2_output.txt');


%% 定点化并转二进制数
% 前向推理仿真结果
feature_r2_bin = str2double(features2bin(feature_r2));
feature_m2_bin = str2double(features2bin(feature_m2));

% 结果保存为txt
mat2txt('feature_r2_bin.txt', feature_r2_bin);
mat2txt('feature_m2_bin.txt', feature_m2_bin);

% 导入数据（一维）
feature_r2_bin_db = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_matlab\feature_r2_bin.txt');
feature_m2_bin_db = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_matlab\feature_m2_bin.txt');


%% 异或对比
xor_compare_r2 = xor(conv2_output, feature_r2_bin_db);
xor_compare_m2 = xor(pool2_output, feature_m2_bin_db);

%% 正确结果统计
r2_count = sum(xor_compare_r2==0)
m2_count = sum(xor_compare_m2==0)

r2_count_acc = sum(xor_compare_r2==0)/length(xor_compare_r2) * 100
m2_count_acc = sum(xor_compare_m2==0)/length(xor_compare_m2) * 100

% 结果保存为txt
file_1 = fopen('xor_compare_r2.txt', 'a');
fprintf(file_1, '%d\n', xor_compare_r2);
fprintf(file_1, 'r2_count = %d and r2_count_acc = %.2f', r2_count, r2_count_acc);
fclose(file_1);

file_2 = fopen('xor_compare_m2.txt', 'a');
fprintf(file_2, '%d\n', xor_compare_m2);
fprintf(file_2, 'm2_count = %d and m2_count_acc = %.2f', m2_count, m2_count_acc);
fclose(file_2);


