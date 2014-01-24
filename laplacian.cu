#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "eigenmap.h"
#include "book.h"
#include <matio.h>

__global__ void diag(double *dev_d, const double *dev_w, int n_patch);
__global__ void compute_l(double *dev_w, int n_patch);

/*
 * laplacian computes the Laplacian matrix based on the weight matrix.
 *
 * dev_w: the weight matrix
 * n_patch: the dimension of dev_w and dev_l
 * Note: the Laplacian matrix is computed in-place and overwrites dev_w.
 */
void laplacian(double *dev_w, int n_patch)
{
    cublasHandle_t handle;
    double *dev_d;

    cublasCreate(&handle);

    HANDLE_ERROR(cudaMalloc((void **)&dev_d, n_patch * sizeof(double)));
    HANDLE_ERROR(cudaMemset(dev_d, 0, n_patch * sizeof(double)));

    // Compute diagonal matrix
    diag<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dev_d, dev_w, n_patch);

    // W <- D^(-1/2) * W * D^(-1/2)
    cublasDdgmm(handle, CUBLAS_SIDE_LEFT, n_patch, n_patch, dev_w, n_patch,
                dev_d, 1, dev_w, n_patch);
    cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, n_patch, n_patch, dev_w, n_patch,
                dev_d, 1, dev_w, n_patch);

    // L <- I - W
    compute_l<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dev_w, n_patch);

    HANDLE_ERROR(cudaFree(dev_d));
    cudaDeviceSynchronize();
    cublasDestroy(handle);
}

__global__ void diag(double *dev_d, const double *dev_w, int n_patch)
{
    int b = blockIdx.x;
    int i, j, size; 
    __shared__ double cache[THREADS_PER_BLOCK];

    /* binary reduction computes the sum of b-th column */
    while (b < n_patch){
        size = THREADS_PER_BLOCK / 2;
        i = threadIdx.x;
        j = i;
        cache[i] = 0.0;
        while (j < n_patch) {
            // load partial sums into shared memory
            cache[i] += dev_w[b * n_patch + j];
            j += THREADS_PER_BLOCK; 
        }
        __syncthreads();
        // reduce the shared array into one output
        while(size != 0){
            if (i < size)
                cache[i] += cache[i+size];
            __syncthreads();
            size /= 2;
        }
        if (i == 0)
            dev_d[b] = 1/sqrt(cache[0]);
        __syncthreads();

        b += BLOCKS_PER_GRID;
    }
}
__global__ void compute_l(double *dev_w, int n_patch)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int N = n_patch * n_patch;
    while (tid < N) {
        dev_w[tid] = ((tid % (n_patch + 1) == 0) ? 1.0 : 0.0) - dev_w[tid];
        tid += blockDim.x * gridDim.x;
    }
}
