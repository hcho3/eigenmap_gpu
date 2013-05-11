% pair_weight2() 
%
%       returns the weight between each two patches. 

function w = pair_weight2(patch1, patch2, pars, option)

switch option 
    case 1
        diff1 = sum(sum((patch1.data - patch2.data).^2));
    case 2
        patch1.data = sort(patch1.data(:));
        patch2.data = sort(patch2.data(:));
        diff1 = sum((patch1.data - patch2.data).^2);
end


w1 = exp( -diff1/(numel(patch1.data)*pars(1)^2) );

diff2 = sum((patch1.pos - patch2.pos).^2);

w2 = exp( -diff2/(ndims(patch1.data)*pars(2)^2) );

w = w1*w2;