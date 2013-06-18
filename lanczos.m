function [V, D] = lanczos(A, num_eigs, k)

tic
n = size(A, 1);

Q = zeros(n, k);
alpha = zeros(1, k);
beta = zeros(1, k);
p = zeros(n, 1);

% generate random r0 with norm 1.
R0 = rand(n, 1);
R0 = R0 / norm(R0, 2);

for i=1:k
    if i == 1
        Q(:, i) = R0;
        p = A * Q(:, 1);
    else
        Q(:, i) = p / beta(i - 1);
        p = Q(:, i - 1);
        p = A * Q(:, i) - beta(i - 1) * p;
    end
    alpha(i) = Q(:, i)' * p;
    p = p - alpha(i) * Q(:, i);
    beta(i) = norm(p, 2);
end

% build k-by-k submatrix T that has a similar eigensystem.
T = diag(alpha) + diag(beta(1:end-1), 1) + diag(beta(1:end-1), -1);

[U, D] = eigs(T, num_eigs, 'sm');
% permute D
D = diag(D);
D = D(end:-1:1);
if nargout < 2
    V = D;
else
    V = Q(:, 1:k)*U;
    D = diag(D);
    % permute V 
    V = V(:, end:-1:1);
end
toc

%{ print spectrums
if nargin == 3
    lanczos_spectrum = sort(eig(T));
    original_spectrum = sort(eig(A));
    figure;
    plot(1:n, original_spectrum, 'k.');
    saveas(gcf, sprintf('results/%s/%s_spectrum.eps', str, str), 'eps2c');
    figure;
    plot(1:k, lanczos_spectrum, 'b.');
    saveas(gcf, sprintf('results/%s/%s_lanczos_spectrum.eps', str, str), ...
           'eps2c');
end
%}
