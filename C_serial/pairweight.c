#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "eigenmap.h"

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
void diff_square(int patch0, double *data, double *temp, double *pos, double *ptemp, int scale, int pos_dim, int n_patch);
void reduce(int patch0, double *w, double *temp, double *ptemp, int scale, int pos_dim, int par0, int par1, int n_patch);

void pairweight(double *w, int n_patch, double *data, double *pos, int scale[2], int pos_dim, int par[2], int option)
{
	
	double *temp, *ptemp;
	int j;

	temp = (double *)malloc(scale[0]*scale[1]*n_patch*sizeof(double));
	ptemp = (double *)malloc(pos_dim*n_patch*sizeof(double));

	for (j=1; j<n_patch; j++){
		diff_square(j, data, temp, pos, ptemp, scale[0]*scale[1], pos_dim, n_patch);
		reduce(j, w, temp, ptemp, scale[0] * scale[1], pos_dim, par[0], par[1], n_patch);
	}

	

	free(temp);
	free(ptemp);
}

void diff_square(int patch0, double *data, double *temp, double *pos, double *ptemp, int scale, int pos_dim, int n_patch)
{
	int patch1;
	int i;
	for (patch1=0; patch1<patch0; patch1++){
		for (i=0; i<scale; i++){
			temp[patch1 * scale + i] = (data[patch1 * scale + i] - data[patch0 * scale + i]) * (data[patch1 * scale + i] - data[patch0 * scale + i]);
		}
		for (i=0; i<pos_dim; i++){
			ptemp[patch1 * pos_dim + i] = (pos[patch1 * pos_dim + i] - pos[patch0 * pos_dim + i]) * (pos[patch1 * pos_dim + i] - pos[patch0 * pos_dim + i]);
		}
	}
}

void reduce(int patch0, double *w, double *temp, double *ptemp, int scale, int pos_dim, int par0, int par1, int n_patch)
{
	int patch1;
	int i;
	double diff;
	for (patch1=0; patch1 < patch0; patch1++){
		diff = 0.0;
		for (i=0; i<scale; i++)
			diff += temp[patch1 * scale + i];
		w[patch1 + patch0 * n_patch] = exp(-diff/(scale * par0 * par0));
		diff = 0.0;
		for (i=0; i < pos_dim; i++)
			diff += ptemp[patch1 * pos_dim + i];
		
		w[patch1 + patch0 * n_patch] *= exp(-diff/(pos_dim * par1 * par1));
		w[patch0 + patch1 * n_patch] = w[patch1 + patch0 * n_patch];
	}
}