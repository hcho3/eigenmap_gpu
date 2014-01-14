#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <lapacke.h>
#include <string.h>

void eigs(double *F, double *Es, double *l, int n_eigs, int n_patch)
{
	double *eigenvalue = (double *)malloc(n_patch*sizeof(double));
	lapack_int n = n_patch;
	lapack_int info;

	printf("n_patch = %d\n", n);
	info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'L', n, l, n, eigenvalue);
	printf("info = %d\n", info);
	memcpy(F, l, n_eigs*n_patch*sizeof(double));
	memcpy(Es, eigenvalue, n_eigs*sizeof(double));
	free(eigenvalue);
}
