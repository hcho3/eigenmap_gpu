function bootstrap_vanilla(str, par1, par2, num_it)
% test multiscale Laplacian manifold learning

tic

scale = [4, 4];
addpath('./Test_Data');
color2d = imread(sprintf('./Test_Data/%s.jpg', str));
gray2d = rgb2gray(color2d);

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

%imshow( uint8(patches(:,:,1)) );

%% construct weight matrix among the patches

pars = [10, 50];
W = zeros(n_patch);

matlabpool

for i = 1:n_patch
    fprintf('i = %d out of %d\n', i, n_patch);
    parfor j = i+1:n_patch
        patch1 = struct('data', patches.data(:, :, i), 'pos', patches.pos(:, i));
        patch2 = struct('data', patches.data(:, :, j), 'pos', patches.pos(:, j));
        W(i,j) = pair_weight2(patch1, patch2, pars, 1);
    end
end

W = W + W';

%% perform spectral clustering
D = diag(sum(W));
NUM_EIGS = 3;
L = eye(n_patch) - D^(-1/2)*W*D^(-1/2);
[F,Es] = lanczos(L,NUM_EIGS,num_it);
save(sprintf('L_%s.mat', str), 'L');

% class = kmeans(F(:,2:3),2);
% figure; group1 = find(class==1); plot(F(group1,2),F(group1,3),'.');
% hold on; group2 = find(class==2); plot(F(group2,2),F(group2,3),'.r')

Es = diag(Es);
F = diff_map(Es,F,NUM_EIGS,1);

%class = kmeans(F(:,2:3),2);
%figure; group1 = find(class==1); plot(F(group1,2),F(group1,3),'.');
%hold on; group2 = find(class==2); plot(F(group2,2),F(group2,3),'.r')

th = 0e-3;
group = find(F(:,2)>th);
display_segment(gray2d,scale,group);
saveas(gcf, sprintf('results/%s/%s_%d_%d_it%d_lanczos.eps', str, str, par1, par2, num_it), 'eps2c');
matlabpool close
