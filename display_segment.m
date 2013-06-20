% display_segment()
% 
%       display the indicated segment for the image

function display_segment(test_image, scale, index)

seg = zeros(size(test_image));
ncol = length(test_image)/scale(2);
for n = 1:length(index)-1
    i = ceil(index(n)/ncol);
    j = mod(index(n)-1,ncol) + 1;
    x_range = (i-1)*scale(1)+1: i*scale(1);
    y_range = (j-1)*scale(2)+1: j*scale(2);
    %seg(x_range, y_range) = test_image(x_range, y_range);
    seg(x_range, y_range) = 255;
end

set(0,'DefaultFigureMenu','none');
figure;
imshow(uint8(seg), 'Border', 'tight');
