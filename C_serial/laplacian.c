#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "eigenmap.h"
#include <cblas.h>
#include <memory.h>

void diag(double *d, double *w, int n_patch);
void eye(double *l, int n_patch);

void laplacian(double *l, double *w, int n_patch)
{
	double *d = (double *)malloc(n_patch*n_patch*sizeof(double));
	double *tempw = (double *)malloc(n_patch*n_patch*sizeof(double));

	memset(d, 0, n_patch*n_patch*sizeof(double));
	memset(l, 0, n_patch*n_patch*sizeof(double));
	
	diag(d, w, n_patch);
	eye(l, n_patch);
	cblas_dsymm(CblasColMajor, CblasLeft, CblasLower, n_patch, n_patch, -1.0, d, n_patch, w, n_patch, 0.0, tempw, n_patch);
	cblas_dsymm(CblasColMajor, CblasRight, CblasLower, n_patch, n_patch, 1.0, d, n_patch, tempw, n_patch, 1.0, l, n_patch);
	
	free(d);
	free(tempw);
}

void diag(double *d, double *w, int n_patch)
{
	int i, j;
	double sum;
	for (j=0; j<n_patch; j++){
		sum = 0;
		for (i=0; i<n_patch; i++)
			sum += w[i + j*n_patch];
		d[j + j * n_patch] = 1/sqrt(sum);
	}
}

void eye(double *l, int n_patch)
{
	int i;
	for (i=0; i<n_patch; i++)
		l[i + i*n_patch] = 1.0;
}
