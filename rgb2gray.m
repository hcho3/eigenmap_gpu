function gray=rgb2gray(K)
R=double(K(:,:,1));              % red plane
G=double(K(:,:,2));              % green plane
B=double(K(:,:,3));              % blue plane
gray = zeros(size(R));
row = size(K, 1);
col = size(K, 2);
for x=1:1:row
    for y=1:1:col
        gray(x,y)=(R(x,y)+G(x,y)+B(x,y))/3;   % method1
    end
end