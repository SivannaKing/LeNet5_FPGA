function feature_m = maxpooling_my(feature_r)
[R, C, N] = size(feature_r);
R_m = R/2;C_m = C/2;
feature_m = zeros(R_m, C_m, N);
tmp = zeros(2,2);
for n = 1 : N
    for r = 1 : R_m
        for c = 1 : C_m
            tmp(1,1) = feature_r(r*2-1, c*2-1, n);
            tmp(1,2) = feature_r(r*2-1, c*2  , n);
            tmp(2,1) = feature_r(r*2  , c*2-1, n);
            tmp(2,2) = feature_r(r*2  , c*2  , n);
            feature_m(r,c,n) = max(max(tmp));
        end
    end
end