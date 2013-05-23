#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "book.h"
#include <cuda_runtime.h>
#include "eigenmap.h"

__global__ void diff_reduce(double *dev_w, double *data, double *pos, int scale, int pos_dim, int par0, int par1, int n_patch);

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
    const dim3 grid_size((n_patch + 15) / 16, (n_patch + 15) / 16, 1);
    const dim3 block_size(16, 16, 1);
	
	HANDLE_ERROR(cudaMalloc((void **)&(temp), scale[0]*scale[1]*n_patch*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&(ptemp), pos_dim*n_patch*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&(dev_data), scale[0] * scale[1] * n_patch * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&dev_pos, pos_dim * n_patch * sizeof(double)));

	HANDLE_ERROR(cudaMemcpy(dev_data, data, scale[0] * scale[1] * n_patch * sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_pos, pos, pos_dim * n_patch * sizeof(double), cudaMemcpyHostToDevice));

    diff_reduce<<<grid_size, block_size>>>(dev_w, dev_data, dev_pos, scale[0] * scale[1], pos_dim, par[0], par[1], n_patch);

	HANDLE_ERROR(cudaFree(temp));
	HANDLE_ERROR(cudaFree(ptemp));
	HANDLE_ERROR(cudaFree(dev_data));
	HANDLE_ERROR(cudaFree(dev_pos));
}

__global__ void diff_reduce(double *dev_w, double *data, double *pos, int scale, int pos_dim, int par0, int par1, int n_patch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    double data_dist = 0.0;
    double pos_dist = 0.0;
    int data_offx = x * scale;
    int data_offy = y * scale;
    int pos_offx = x * pos_dim;
    int pos_offy = y * pos_dim;
    double datax, datay, posx, posy;
    int i;

    /*
    W_xy = f( sum((patch_x.data - patch_y.data)^2) )
            * g( sum((patch_x.pos - patch_y.pos)^2) )
    where
    f(x) = exp( -x / (scale * par0 * par0) )
    g(x) = exp( -x / (pos_dim * par1 * par1) )
    */
    if (x == y || x >= n_patch || y >= n_patch)
        return;

    for (i = 0; i < scale; i++) {
        datax = data[data_offx + i];
        datay = data[data_offy + i];
        data_dist += (datax - datay) * (datax - datay);
    }

    for (i = 0; i < pos_dim; i++) {
        posx = pos[pos_offx + i];
        posy = pos[pos_offy + i];
        pos_dist += (posx - posy) * (posx - posy);
    }

    dev_w[x + y * n_patch] = exp( -data_dist / (scale * par0 * par0))
                              * exp( -pos_dist / (pos_dim * par1 * par1));
}
