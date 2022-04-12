%% ����������м��������
conv2_output = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_matlab\graph3\conv2_output.txt');
pool2_output = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_matlab\graph3\pool2_output.txt');


%% ���㻯��ת��������
% ǰ�����������
feature_r2_bin = str2double(features2bin(feature_r2));
feature_m2_bin = str2double(features2bin(feature_m2));

% �������Ϊtxt
mat2txt('feature_r2_bin.txt', feature_r2_bin);
mat2txt('feature_m2_bin.txt', feature_m2_bin);

% �������ݣ�һά��
feature_r2_bin_db = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_matlab\feature_r2_bin.txt');
feature_m2_bin_db = load('E:\20210301_MNIST_LeNet5_Accelerator\forward_simulation_matlab\feature_m2_bin.txt');


%% ���Ա�
xor_compare_r2 = xor(conv2_output, feature_r2_bin_db);
xor_compare_m2 = xor(pool2_output, feature_m2_bin_db);

%% ��ȷ���ͳ��
r2_count = sum(xor_compare_r2==0)
m2_count = sum(xor_compare_m2==0)

r2_count_acc = sum(xor_compare_r2==0)/length(xor_compare_r2) * 100
m2_count_acc = sum(xor_compare_m2==0)/length(xor_compare_m2) * 100

% �������Ϊtxt
file_1 = fopen('xor_compare_r2.txt', 'a');
fprintf(file_1, '%d\n', xor_compare_r2);
fprintf(file_1, 'r2_count = %d and r2_count_acc = %.2f', r2_count, r2_count_acc);
fclose(file_1);

file_2 = fopen('xor_compare_m2.txt', 'a');
fprintf(file_2, '%d\n', xor_compare_m2);
fprintf(file_2, 'm2_count = %d and m2_count_acc = %.2f', m2_count, m2_count_acc);
fclose(file_2);


