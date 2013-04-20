function bootstrap_gpu(str, par1, par2)
% test multiscale Laplacian manifold learning

tic
scale = [4, 4];
addpath('./Test_Data');
color2d = imread(sprintf('./Test_Data/%s.jpg', str));
if size(color2d, 3) > 1
    gray2d = rgb2gray(color2d);
else
    gray2d = color2d;
end

image_size = size(gray2d);
[M,N] = deal(image_size(1), image_size(2));
[m,n] = deal(M/scale(1), N/scale(2));
n_patch = m*n;
patches.data = zeros(scale(1), scale(2), n_patch);
patches.pos = zeros(2, n_patch);

for i = 1:m
    for j = 1:n
        idx = (i-1)*n + j;
        x_range = (i-1)*scale(1)+1: i*scale(1);
        y_range = (j-1)*scale(2)+1: j*scale(2);
        patches.data(:,:,idx) = gray2d(x_range, y_range);
        patches.pos(:,idx) = [(i-1)*scale(1)+1, (j-1)*scale(2)+1]';
    end
end

save(sprintf('%s.mat', str), 'patches');

%imshow( uint8(patches(:,:,1)) );

%% construct weight matrix among the patches
NUM_EIGS = 15;
[status, ~] = system(sprintf('./eigenmap %s.mat %d %d %d', str, NUM_EIGS, par1, par2), '-echo');
if status > 0
    return
end
load('F.mat');
load('Es.mat');
F = diff_map(Es,F,NUM_EIGS,1);

%class = kmeans(F(:,2),2);
%figure; group1 = find(class==1); plot(F(group1,2),F(group1,3),'.');
%hold on; group2 = find(class==2); plot(F(group2,2),F(group2,3),'.r')

th = 0e-3;
group = find(F(:,2)<=th);
display_segment(gray2d,scale,group);
saveas(gcf, sprintf('results/%s/%s_%d_%d_gpu.eps', str, str, par1, par2), 'eps2c');

%quit

toc
