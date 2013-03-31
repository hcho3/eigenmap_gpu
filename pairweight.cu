#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "book.h"
#include <cuda_runtime.h>
#include "eigenmap.h"

__global__ void diff_square(int patch0, double *data, double *temp, double *pos, double *ptemp, int scale, int pos_dim, int n_patch);
__global__ void reduce(int patch0, double *dev_w, double *temp, double *ptemp, int scale, int pos_dim, int par0, int par1, int n_patch);

/* pairweight calculates and modifies the weight matrix dev_w (symmetric)
 * dev_w: device pointer to allocated space for the symmetric weight matrix
 * data: the data field in patches
 * pos: the pos field in patches
 * n_patch: number of patches in patches
 * scale[0] * scale[1] : size of data field for each patch
 * pos_dim: size of pos field for each patch
 * par[2]: parameters
 * option: option (not implemented so far)
 */

/* ---- corresponding Matlab code ----
 * function w = pair_weight2(patch1, patch2, pars, option)
	% diff_square
	temp = (patch1.data - patch2.data).^2
	ptemp = (patch1.pos - patch2.pos).^2

	% reduce
	diff1 = sum(sum(temp));
	diff2 = sum(ptemp);
	w1 = exp( -diff1/(numel(patch1.data)*pars(1)^2) );
	w2 = exp( -diff2/(ndims(patch1.data)*pars(2)^2) );
	w = w1*w2;
 */
void pairweight(double *dev_w, int n_patch, double *data, double *pos, int scale[2], int pos_dim, int par[2], int option)
{
	double *temp, *ptemp;
	double *dev_data, *dev_pos;	
	int j;
	
	HANDLE_ERROR(cudaMalloc((void **)&(temp), scale[0]*scale[1]*n_patch*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&(ptemp), pos_dim*n_patch*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&(dev_data), scale[0] * scale[1] * n_patch * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&dev_pos, pos_dim * n_patch * sizeof(double)));

	HANDLE_ERROR(cudaMemcpy(dev_data, data, scale[0] * scale[1] * n_patch * sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_pos, pos, pos_dim * n_patch * sizeof(double), cudaMemcpyHostToDevice));

	for (j = 1; j < n_patch; j++){
		diff_square<<<BPG, TPB>>>(j, dev_data, temp, dev_pos, ptemp, scale[0] * scale[1], pos_dim, n_patch);
		reduce<<<BPG, TPB>>>(j, dev_w, temp, ptemp, scale[0] * scale[1], pos_dim, par[0], par[1], n_patch);
	}

	HANDLE_ERROR(cudaFree(temp));
	HANDLE_ERROR(cudaFree(ptemp));
	HANDLE_ERROR(cudaFree(dev_data));
	HANDLE_ERROR(cudaFree(dev_pos));
}


/*
 * diff_square takes the square of the difference between the data
 * fields and pos fields of two patches (patch0, patch1) and put the
 * resulting difference matrices to temp and ptemp, respectively.
 * Notice that for each invocation,  patch0 is fixed while patch1 is
 * iterated from 0 to patch0 - 1.
 */
__global__ void diff_square(int patch0, double *data, double *temp, double *pos, double *ptemp, int scale, int pos_dim, int n_patch)
{
	/*
		temp = (patch1.data - patch2.data).^2
		ptemp = (patch1.pos - patch2.pos).^2
	*/
	int patch1 = blockIdx.x;
	int i;

	while(patch1 < patch0) {
		i = threadIdx.x;
		while (i < scale) {
			temp[patch1 * scale + i] = (data[patch1 * scale + i] - data[patch0 * scale + i]) * (data[patch1 * scale + i] - data[patch0 * scale + i]);
			i += TPB;
		}
		__syncthreads();
		i = threadIdx.x;
		while (i < pos_dim){
			ptemp[patch1 * pos_dim + i] = (pos[patch1 * pos_dim + i] - pos[patch0 * pos_dim + i]) * (pos[patch1 * pos_dim + i] - pos[patch0 * pos_dim + i]);
			i += TPB;
		}
		__syncthreads();
		patch1 += BPG;
	}
}


/*
 * reduce sums the elements of temp and ptemp, exponentiates each sum, and returns their product.
 * This is done by reduce summation.
 */
__global__ void reduce(int patch0, double *dev_w, double *temp, double *ptemp, int scale, int pos_dim, int par0, int par1, int n_patch)
{
	/*
		diff1 = sum(sum(temp));
		diff2 = sum(ptemp);
		w1 = exp( -diff1/(numel(patch1.data)*pars(1)^2) );
		w2 = exp( -diff2/(ndims(patch1.data)*pars(2)^2) );
		w = w1*w2;
	*/
	int patch1 = blockIdx.x;
	int i, j;
	int size;
	__shared__ double cache[TPB];

	while (patch1 < patch0){
		size = TPB/2;
		i = threadIdx.x; // thread index
		j = i;		     // loop variable
		cache[i] = 0.0;
		while (j < scale) {
			cache[i] += temp[patch1 * scale + j];
			j += TPB;		
		}
		__syncthreads();
		while (size != 0){
			if (i < size){
				cache[i] += cache[i+size];
			}
			__syncthreads();
			size /= 2;
		}
		if (i == 0){
			dev_w[patch1 + patch0 * n_patch] = exp(-cache[0]/(scale * par0 * par0));
		}
		__syncthreads();
		
		cache[i] = 0.0;
		size = TPB/2;
		j = i;
		while (j < pos_dim){
			cache[i] += ptemp[patch1 * pos_dim + j];
			j += TPB;
		}
		__syncthreads();
		while (size != 0){
			if (i < size){
				cache[i] += cache[i + size];
			}
			__syncthreads();
			size /= 2;
		}
		if (i == 0) {
			dev_w[patch1 + patch0 * n_patch] *= exp(-cache[0]/(pos_dim * par1 * par1));
			dev_w[patch0 + patch1 * n_patch] = dev_w[patch1 + patch0 * n_patch];
		}
		__syncthreads();

		patch1 += BPG;
	}

}
