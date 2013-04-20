#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "book.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "eigenmap.h"
#include <matio.h>
#include <string>
using std::string;

__global__ void diag(double *dev_d, const double *dev_w, int n_patch);
__global__ void eye(double *dev_l, int n_patch);
__global__ void compute_l(double *dev_l, double *dev_w, int n_patch);
__device__ double atomicAdd(double* address, double val);
void diag_similarity_transform(cublasHandle_t handle, double *dev_w, int n_patch);
void write_diag(const double *dev_d, int n_patch,
                string varname, string filename);
extern void write_laplacian(const double *dev_l, int n_patch,
                            string varname, string filename);
extern string filename;

/*
 * laplacian computes the Laplacian matrix based on the weight matrix.
 * dev_l: the laplacian matrix
 * dev_w: the weight matrix
 * n_patch: the dimension of dev_w and dev_l
 * Note: dev_l and dev_w are overwritten and stay (not freed) in the memory.
 */
void laplacian(double *dev_l, double *dev_w, int n_patch)
{
	/* ---- corresponding Matlab code ----
	 * D = diag(sum(W));
	 * L = eye(n_patch) - D^(-1/2)*W*D^(-1/2);
	 */
	//double *dev_d;
	cublasHandle_t handle;
	//const double alpha = -1.0;
	//const double beta = 0.0;
	//const double one = 1.0;
	
	cublasCreate(&handle);
	//HANDLE_ERROR(cudaMalloc((void **)&dev_d, n_patch * n_patch * sizeof(double)));
	//HANDLE_ERROR(cudaMemset(dev_d, 0, n_patch * n_patch * sizeof(double)));
	HANDLE_ERROR(cudaMemset(dev_l, 0, n_patch * n_patch * sizeof(double)));
	//diag<<<BPG, TPB>>>(dev_d, dev_w, n_patch);

	eye<<<BPG, 1>>>(dev_l, n_patch); // L <- eye(n_patch)

#if 0 /* OLD CODE */
	// W <- (-1) * D * W + 0
    write_laplacian(dev_w, n_patch, "W_gpu_input",
                    filename + "_gpu_laplacian_input.mat");
	cublasDsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, n_patch, n_patch, &alpha, dev_d, n_patch, dev_w, n_patch, &beta, dev_w, n_patch);

    write_laplacian(dev_w, n_patch, "W_gpu_intermediate",
                    filename + "_gpu_laplacian_intermediate.mat");
	// L <- 1 * W * D + L
	cublasDsymm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, n_patch, n_patch, &one, dev_d, n_patch, dev_w, n_patch, &one, dev_l, n_patch);
#endif
    /* NEW */
    // W <- D * W * D
    write_laplacian(dev_w, n_patch, "W_gpu_input",
                    filename + "_gpu_laplacian_input.mat");
    diag_similarity_transform(handle, dev_w, n_patch);
    // L <- L - W
    compute_l<<<BPG, TPB>>>(dev_l, dev_w, n_patch);
    cudaDeviceSynchronize();
	
#if 0
	//DEBUG
	double *l;
	int i, j;
	l = (double *)malloc(n_patch * n_patch * sizeof(double));
	HANDLE_ERROR(cudaMemcpy(l, dev_l, n_patch * n_patch * sizeof(double), cudaMemcpyDeviceToHost));
	for (i = 0; i < 15; i++) {
		for (j = 0; j < 15; j++)
			printf("%8.6f ", l[i + j * n_patch]);
		printf("\n");
	}
	free(l);
#endif
	//HANDLE_ERROR(cudaFree(dev_d));
	cublasDestroy(handle);
}

void diag_similarity_transform(cublasHandle_t handle, double *dev_w, int n_patch)
{
    double *dev_d;
	HANDLE_ERROR(cudaMalloc((void **)&dev_d, n_patch * sizeof(double)));
	HANDLE_ERROR(cudaMemset(dev_d, 0, n_patch * sizeof(double)));

	diag<<<BPG, TPB>>>(dev_d, dev_w, n_patch);
    cudaDeviceSynchronize();
    write_diag(dev_d, n_patch, "diag_gpu", filename + "_gpu_diag.mat");

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

/*
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
			dev_d[b + b*n_patch] = 1/sqrt(cache[0]); // raise each diagonal entry to power of -1/2
		}
		__syncthreads();

		b += BPG;
	}
}
*/
__global__ void eye(double *dev_l, int n_patch)
{
	int i = blockIdx.x;
	while(i < n_patch) {
		dev_l[i + i * n_patch] = 1.0;
		i += BPG;
	}
}
__global__ void compute_l(double *dev_l, double *dev_w, int n_patch)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int N = n_patch * n_patch;
    while (tid < N) {
        dev_l[tid] -= dev_w[tid];
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
void write_diag(const double *dev_d, int n_patch,
                string varname, string filename)
{
    mat_t *matfp;
    matvar_t *L;
    double *host_d = (double *)malloc(n_patch * sizeof(double));
	size_t d_dims[1] = {n_patch};

	matfp = Mat_CreateVer(filename.c_str(), NULL, MAT_FT_DEFAULT);

    cudaMemcpy(host_d, dev_d, n_patch * sizeof(double), cudaMemcpyDeviceToHost);
	L = Mat_VarCreate(varname.c_str(), MAT_C_DOUBLE, MAT_T_DOUBLE, 1, d_dims, host_d, 0);
    Mat_VarWrite(matfp, L, MAT_COMPRESSION_NONE);
    
	Mat_Close(matfp);
    Mat_VarFree(L);
    free(host_d);
}
