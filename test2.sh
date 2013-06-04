#!/bin/bash
thread[0]=16
thread[1]=32
thread[2]=64
thread[3]=256
thread[4]=512
thread[5]=1024
block[0]=16384
block[1]=8192
block[2]=4096
block[3]=1024
block[4]=512
block[5]=128
for i in {0..5}
do
    echo "TPB=${thread[$i]}, BPG=${block[$i]}"
    nvcc -arch=sm_35 -rdc=true -c laplacian.cu -o laplacian.o -Xcompiler -fPIC -I/home/hcho3/include -D TPB=${thread[$i]} -D BPG=${block[$i]}
    nvcc -arch=sm_35 -rdc=true -c lanczos.cu -o lanczos.o -Xcompiler -fPIC -I/home/hcho3/include -D TPB=${thread[$i]} -D BPG=${block[$i]}
    nvcc -arch=sm_35 -rdc=true -o eigenmap eigenmap.o pairweight.o laplacian.o book.o lanczos.o /home/hcho3/lib/libmatio.a /usr/local/magma/lib/libmagma.a /usr/local/magma/lib/libmagmablas.a -llapack -lblas -lm -lz -L/usr/local/cuda/lib64 -lcublas -lcudadevrt
    ./eigenmap nCPM9_large.mat 3 250 10 50
    echo ""
done
