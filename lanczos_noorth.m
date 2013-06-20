function [V, D] = lanczos_noorth(A, num_eigs, k)

tic
n = size(A, 1);

Q = zeros(n, k + 1);
alpha = zeros(1, k);
beta = zeros(1, k);
z = zeros(n, 1);

% generate random b with norm 1.
b = rand(n, 1);
b = b / norm(b, 2);

Q(:, 1) = b;

for i=1:k
    z = A * Q(:, i);
    alpha(i) = Q(:, i)' * z;
    if i == 1
        z = z - alpha(i) * Q(:, i);
    else
        z = z - alpha(i) * Q(:, i) - beta(i - 1) * Q(:, i - 1);
    end

    beta(i) = norm(z, 2);
    Q(:, i + 1) = z / beta(i);
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

%{
% print spectrums
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
