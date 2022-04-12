%% 导入加速器中间层输出结果
fc1_output = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_matlab\graph4\fc1_output.txt');

%% 定点化并转二进制数
% 前向推理仿真结果
feature_fc1_r1_bin = str2(features2bin(feature_fc1_r1));

% 结果保存为txt
mat2txt('feature_fc1_r1_bin.txt', feature_fc1_r1_bin);

% 导入数据（一维）
feature_fc1_r1_bin_db = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_matlab\feature_fc1_r1_bin.txt');


%% 异或对比
xor_compare_fc1_r1 = xor(fc1_output, feature_fc1_r1_bin_db);

%% 正确结果统计
fc1_r1_count = sum(xor_compare_fc1_r1==0)

fc1_r1_count_acc = fc1_r1_count/length(xor_compare_fc1_r1) * 100

% 结果保存为txt
file_1 = fopen('xor_compare_fc1_r1.txt', 'a');
fprintf(file_1, '%d\n', xor_compare_fc1_r1);
fprintf(file_1, 'fc1_r1_count = %d and fc1_r1_count_acc = %.2f', fc1_r1_count, fc1_r1_count_acc);
fclose(file_1);
