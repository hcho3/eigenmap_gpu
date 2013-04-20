for i=5:5
    fprintf(1,    'Processing nCPM%d.jpg ...\n', i);
    bootstrap_gpu(sprintf('nCPM%d', i), 10, 50);
%   fprintf(1,    'Processing nCPM%d.jpg ...\n', i);
%   bootstrap_gpu(sprintf('nCPM%d', i), 10, 50);
end
%system('python notify.py');
