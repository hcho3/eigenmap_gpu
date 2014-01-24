#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "book.h"
#include <cuda_runtime.h>
#include "eigenmap.h"

__global__ void diff_reduce(double *dev_w, double *feat, double *pos,
                            int feat_dim, int pos_dim, int par0,
                            int par1, int n_patch);

/* pairweight calculates and modifies the weight matrix dev_w (symmetric)
 *
 * dev_w: device pointer to allocated space for the symmetric weight matrix
 * feat: list of features vectors
 * pos: list of position vectors
 * n_patch: number of patches
 * feat_dim : dimension of each features vector
 * pos_dim: dimension of each position vector
 * par[2]: parameters
 * option: option (not implemented so far)
 */
void pairweight(double *dev_w, int n_patch, double *feat, double *pos,
                int feat_dim[2], int pos_dim, int par[2], int option)
{
    double *dev_feat, *dev_pos; 
    const dim3 grid_size((n_patch + 15) / 16, (n_patch + 15) / 16, 1);
    const dim3 block_size(16, 16, 1);
    
    HANDLE_ERROR(cudaMalloc((void **)&dev_feat, feat_dim[0] * feat_dim[1] *
                            n_patch * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_pos, pos_dim * n_patch *
                            sizeof(double)));

    HANDLE_ERROR(cudaMemcpy(dev_feat, feat, feat_dim[0] * feat_dim[1] *
                            n_patch * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_pos, pos, pos_dim * n_patch * sizeof(double),
                            cudaMemcpyHostToDevice));

    diff_reduce<<<grid_size, block_size>>>(dev_w, dev_feat, dev_pos,
        feat_dim[0] * feat_dim[1], pos_dim, par[0], par[1], n_patch);

    HANDLE_ERROR(cudaFree(dev_feat));
    HANDLE_ERROR(cudaFree(dev_pos));
}

__global__ void diff_reduce(double *dev_w, double *feat, double *pos,
                            int feat_dim, int pos_dim, int par0,
                            int par1, int n_patch)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    double feat_dist = 0.0; // running entry sum of d_ij
    double pos_dist = 0.0;  // running entry sum of f_ij
    int feat_offi = i * feat_dim; // offset of x_i
    int feat_offj = j * feat_dim; // offset of x_j
    int pos_offi = i * pos_dim;   // offset of p_i
    int pos_offj = j * pos_dim;   // offset of p_j
    double feat_i, feat_j, pos_i, pos_j;
    // temporary local variables for entry sum calculation
    int k;

    if (i == j || i >= n_patch || j >= n_patch)
        return;

    /* thread (i, j) computes W_ij */

    // get the k-th element of difference vector d_ij
    // and add it to feat_dist
    for (k = 0; k < feat_dim; k++) {
        feat_i = feat[feat_offi + k];
        feat_j = feat[feat_offj + k];
        feat_dist += (feat_i - feat_j) * (feat_i - feat_j);
    }

    // get the k-th element of difference vector f_ij
    // and add it to pos_dist
    for (k = 0; k < pos_dim; k++) {
        pos_i = pos[pos_offi + k];
        pos_j = pos[pos_offj + k];
        pos_dist += (pos_i - pos_j) * (pos_i - pos_j);
    }

    dev_w[i + j * n_patch]
        = exp( -feat_dist / (feat_dim * par0 * par0))
           * exp( -pos_dist / (pos_dim * par1 * par1));
}
