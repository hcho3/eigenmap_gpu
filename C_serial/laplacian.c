#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "eigenmap.h"
#include <cblas.h>
#include <memory.h>

static void diag(double *d, const double *w, int n_patch);
static void compute_l(double *w, int n_patch);

static void diag_similarity_transform(double *w, int n_patch);

/*
 * laplacian computes the Laplacian matrix based on the weight matrix.
 *
 * w: the weight matrix
 * n_patch: the dimension of dev_w and dev_l
 * Note: the Laplacian matrix is computed in-place and overwrites w.
 */
void laplacian(double *w, int n_patch)
{
	/* ---- corresponding Matlab code ----
	 * D = diag(sum(W));
	 * L = eye(n_patch) - D^(-1/2)*W*D^(-1/2);
	 */

    // W <- D^(-1/2) * W * D^(-1/2)
    diag_similarity_transform(w, n_patch);
    // L <- I - W
    compute_l(w, n_patch);
}

static void diag_similarity_transform(double *w, int n_patch)
{
    double *d = (double *)calloc(n_patch, sizeof(double));
    int i;

    diag(d, w, n_patch);

    // row operations
    for (i = 0; i < n_patch; i++)
        cblas_dscal(n_patch, d[i], &w[i], n_patch);
    // column operations
    for (i = 0; i < n_patch; i++)
        cblas_dscal(n_patch, d[i], &w[i * n_patch], 1);

    free(d);
}

static void diag(double *d, const double *w, int n_patch)
{
    int i, j;
    double sum;
    for (j = 0; j < n_patch; j++){
        sum = 0;
        for (i = 0; i < n_patch; i++)
            sum += w[i + j * n_patch];
        d[j] = 1 / sqrt(sum);
    }   
}

static void compute_l(double *w, int n_patch)
{
    int N = n_patch * n_patch;
    int i;

    for (i = 0; i < N; i++)
        w[i] = ((i % (n_patch + 1) == 0) ? 1.0 : 0.0) - w[i];
}
