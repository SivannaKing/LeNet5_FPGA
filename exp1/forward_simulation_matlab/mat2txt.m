function back = mat2txt(file_Name, matrix)
% MAT2TXT��ʵ�ֽ�.mat��׺�ľ���matrix����������׺���ļ�
% ת���� .txt ������mat2txt( 'filename.txt', data );
fop = fopen( file_Name, 'w' );
[H, W, C] = size(matrix);
for c = 1:C
    for h = 1:H
        for w = 1:W
            fprintf(fop, '%s', strtrim(mat2str( matrix(h, w, c))));
            fprintf(fop, '\n' );
        end
    end
end
back = fclose( fop ) ;



