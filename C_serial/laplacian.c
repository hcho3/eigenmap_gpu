#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "eigenmap.h"
#include <cblas.h>
#include <matio.h>

#include <memory.h>

void diag(double *d, double *w, int n_patch);
void eye(double *l, int n_patch);
void write_laplacian(double *l, int n_patch,
                     const char *varname, const char *filename);
extern char filename[];

void laplacian(double *l, double *w, int n_patch)
{
	double *d = (double *)malloc(n_patch*n_patch*sizeof(double));
	double *tempw = (double *)malloc(n_patch*n_patch*sizeof(double));
    char *tmpstr = (char *)malloc(BUFSIZ * sizeof(char));

	memset(d, 0, n_patch*n_patch*sizeof(double));
	memset(l, 0, n_patch*n_patch*sizeof(double));
	
	diag(d, w, n_patch);
	eye(l, n_patch);
    sprintf(tmpstr, "%s_c_laplacian_input.mat", filename);
    write_laplacian(w, n_patch, "W_c_input", tmpstr);
	cblas_dsymm(CblasColMajor, CblasLeft, CblasLower, n_patch, n_patch, -1.0, d, n_patch, w, n_patch, 0.0, tempw, n_patch);
	/*
	int i, j;
	printf("debug...\n");
	for (i = 1000; i < 1015; i++) {
		for (j = 1000; j < 1015; j++)
			printf("%8.6f ", tempw[i + j * n_patch]);
		printf("\n");
	}
	printf("debug...\n");*/
    sprintf(tmpstr, "%s_c_laplacian_intermediate.mat", filename);
    write_laplacian(tempw, n_patch, "W_c_intermediate", tmpstr);
	cblas_dsymm(CblasColMajor, CblasRight, CblasLower, n_patch, n_patch, 1.0, d, n_patch, tempw, n_patch, 1.0, l, n_patch);
    sprintf(tmpstr, "%s_c_laplacian_final.mat", filename);
    write_laplacian(l, n_patch, "L_c_final", tmpstr);
	
    free(tmpstr);

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


void write_laplacian(double *host_l, int n_patch,
                     const char *varname, const char *filename)
{
    mat_t *matfp;
    matvar_t *L;
	size_t L_dims[2] = {n_patch, n_patch};

	matfp = Mat_CreateVer(filename, NULL, MAT_FT_DEFAULT);

	L = Mat_VarCreate(varname, MAT_C_DOUBLE, MAT_T_DOUBLE, 2, L_dims, host_l, 0);
    Mat_VarWrite(matfp, L, MAT_COMPRESSION_NONE);
    
	Mat_Close(matfp);
    Mat_VarFree(L);
}
