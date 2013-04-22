#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "book.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "eigenmap.h"
#include <matio.h>

__global__ void diag(double *dev_d, const double *dev_w, int n_patch);
__global__ void compute_l(double *dev_w, int n_patch);
__device__ double atomicAdd(double* address, double val);
void diag_similarity_transform(cublasHandle_t handle, double *dev_w, int n_patch);

/*
 * laplacian computes the Laplacian matrix based on the weight matrix.
 * dev_w: the weight matrix
 * n_patch: the dimension of dev_w and dev_l
 * Note: the Laplacian matrix is computed in-place and overwrites dev_w.
 */
void laplacian(double *dev_w, int n_patch)
{
	/* ---- corresponding Matlab code ----
	 * D = diag(sum(W));
	 * L = eye(n_patch) - D^(-1/2)*W*D^(-1/2);
	 */
	cublasHandle_t handle;
	
	cublasCreate(&handle);

    // W <- D * W * D
    diag_similarity_transform(handle, dev_w, n_patch);
    // L <- I - W
    compute_l<<<BPG, TPB>>>(dev_w, n_patch);
    cudaDeviceSynchronize();
	
	cublasDestroy(handle);
}

void diag_similarity_transform(cublasHandle_t handle, double *dev_w, int n_patch)
{
    double *dev_d;
	HANDLE_ERROR(cudaMalloc((void **)&dev_d, n_patch * sizeof(double)));
	HANDLE_ERROR(cudaMemset(dev_d, 0, n_patch * sizeof(double)));

	diag<<<BPG, TPB>>>(dev_d, dev_w, n_patch);
    cudaDeviceSynchronize();

    // row operations
    cublasDdgmm(handle, CUBLAS_SIDE_LEFT, n_patch, n_patch, dev_w, n_patch, dev_d, 1, dev_w, n_patch);
    cudaDeviceSynchronize();
    // column operations
    cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, n_patch, n_patch, dev_w, n_patch, dev_d, 1, dev_w, n_patch);
    cudaDeviceSynchronize();

    HANDLE_ERROR(cudaFree(dev_d));
}

__global__ void diag(double *dev_d, const double *dev_w, int n_patch)
{
	int b = blockIdx.x;
	int i, j;
	int size;
	__shared__ double cache[TPB];

	while (b < n_patch){
		size = TPB/2;
		i = threadIdx.x;
		j = i;
		cache[i] = 0.0;
		while (j < n_patch) {
			cache[i] += dev_w[b * n_patch + j];
			j += TPB;	
		}
		__syncthreads();
		while(size != 0){
			if (i < size){
				cache[i] += cache[i+size];
			}
			__syncthreads();
			size /= 2;
		}
		if (i == 0){
			dev_d[b] = 1/sqrt(cache[0]); // raise each diagonal entry to power of -1/2
		}
		__syncthreads();

		b += BPG;
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

__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
