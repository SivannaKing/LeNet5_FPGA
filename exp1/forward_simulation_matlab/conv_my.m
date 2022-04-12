function feature = conv_my(img, weight, bias, N, M, K, O_Size)
% img：输入图像矩阵
% weight：权重矩阵
% bias：偏置矩阵
% N：输入通道数量
% M：输出通道数量
% K：卷积核大小
% O_Size：输出图像尺寸

feature = zeros(O_Size, O_Size, M);
for m = 1 : M
    for r = 1 : O_Size
        for c = 1 : O_Size
            tmp = 0;
            for n = 1 : N
                for i = 1 : K
                    for j = 1 : K
                        tmp = tmp + img(r+i-1, c+j-1, n) * weight(K*K*N*(m-1) + K*K*(n -1) + K*(i-1) + j);
                    end
                end
            end
            feature(r, c, m) = tmp + bias(m);
        end
     end
end

