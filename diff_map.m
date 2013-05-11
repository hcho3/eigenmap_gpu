% calculate the diffusion map

function datamap = diff_map(Es, F, k, t)

N = length(F);
Es = Es.^t;

datamap = zeros(N,k);

for i = 1:k
    datamap(:,i) = Es(i)*F(:,i);
end