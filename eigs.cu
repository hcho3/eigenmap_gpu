#include <stdio.h>
#include <stdlib.h>
#include "magma.h"
#include <assert.h>
#include "book.h"

/*
 * eigs computes the smallest n_eigs eigenvalues for dev_l and the corresponding eigenvectors.
 * F: an array (n_patch by n_eigs) to store the eigenvectors
 * Es: an array (1 by n_eigs) to store the eigenvalues
 * dev_l: an array (n_patch by n_patch) representing the Laplacian matrix
 * n_patch: the dimension of dev_l
 *
 * NOTICE: dev_l will be overwritten in the process of computing eigenvectors. Save your work!
 */
/* ---- corresponding Matlab code ----
 * [F, Es] = eigs(L, n_eigs, 'sm')
 */

extern "C" magma_int_t
magma_dsyevdx(char jobz, char range, char uplo,
              magma_int_t n,
              double *a, magma_int_t lda,
              double vl, double vu, magma_int_t il, magma_int_t iu,
              magma_int_t *m, double *w,
              double *work, magma_int_t lwork,
              magma_int_t *iwork, magma_int_t liwork,
              magma_int_t *info);

void eigs(double *F, double *Es, double *dev_l, int n_eigs, int n_patch)
{
	magma_int_t info;
	magma_int_t nb, lwork, liwork, ldwa;
	magma_int_t *iwork;
	double *work, *wa;
	double *lambda; /* eigenvalues */ 
    magma_int_t ret;

	/* initialize constants */
	nb = magma_get_dsytrd_nb(n_patch);
	lwork = n_patch * nb + 6 * n_patch + 2 * n_patch * n_patch;
	liwork = 3 + 5 * n_patch;
    ldwa = n_patch;
	
	/* initialize workspaces */
	lambda = (double *)malloc(n_patch*sizeof(double));
	wa = (double *)malloc(n_patch * n_patch * sizeof(double));
	iwork = (magma_int_t *)malloc(liwork * sizeof(magma_int_t));
	work = (double *)malloc(lwork * sizeof(double));

	/* Compute eigenvalues and eigenvectors */
    ret = magma_dsyevd_gpu('V', 'L', n_patch, dev_l, n_patch, lambda, wa, ldwa, work, lwork, iwork, liwork, &info);
    printf("ret = %d, info = %d\n", ret, info);
    assert(MAGMA_SUCCESS == ret);

	/* Copy specified number of eigenvalues */
	memcpy(Es, lambda, n_eigs * sizeof(double));
	/* Copy the corresponding eigenvectors */
	HANDLE_ERROR(cudaMemcpy(F, dev_l, n_eigs * n_patch * sizeof(double), cudaMemcpyDeviceToHost) );
    //memcpy(F, l, n_eigs * n_patch * sizeof(double));

	free(iwork);
	free(work);
	free(wa);
	free(lambda);
}
