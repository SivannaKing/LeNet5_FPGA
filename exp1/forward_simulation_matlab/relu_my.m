function feature_r = relu_my(feature_c)
[R, C, N] = size(feature_c);
feature_r = zeros(R, C, N);
for n = 1 : N
    for r = 1 : R
        for c = 1 : C
            if feature_c(r,c,n) >= 0 
                feature_r(r,c,n) = feature_c(r,c,n);
            end
        end
    end
end