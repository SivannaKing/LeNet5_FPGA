function feature_bin = features2bin(feature)
% R： 行 H
% C： 列 W
% N： 通道 Channel

sign_bit = 1; % 符号位宽
int_bit = 4; % 整数位宽
frac_bit = 3; % 小数位宽
sum_bit = sign_bit + int_bit + frac_bit; % 二进制总位宽

[R,C,N]= size(feature);
feature_bin = strings(R,C,N);
feature = int16(round(feature * pow2(frac_bit)));
for r = 1 : R
    for c = 1 : C
        for n = 1 : N
            a = feature(r, c, n);
            if a >= 0
                tmp = dec2bin(a,sum_bit);
                tmp1 = '0'+ string(tmp);
            else
                tmp= dec2bin(a*-1,sum_bit);
                tmp1 = '1'+ string(tmp);
            end
            feature_bin(r, c, n)=tmp1;
        end
    end
end