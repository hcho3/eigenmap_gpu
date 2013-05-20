#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "eigenmap.h"
#include <cblas.h>
#include <lapacke.h>
#include <matio.h>
#include <memory.h>
#include <time.h>
/*
 * lanczos computes the smallest n_eigs eigenvalues for dev_l and the corresponding eigenvectors using the Lanczos algorithm.
 * F: an array (n_patch by n_eigs) to store the eigenvectors
 * Es: an array (1 by n_eigs) to store the eigenvalues
 * dev_l: an array (n_patch by n_patch) representing the Laplacian matrix
 * n_patch: the dimension of dev_l
 */
/* ---- corresponding Matlab code ----
 * [F, Es] = lanczos(L, n_eigs)
 */
static double norm2(double *v, int length);
static void divide_copy(double *dest, const double *src, int length, const double *divisor);
static void build_tridiagonal(double *T, const double *alpha, const double *beta, int Tdim);

void lanczos(double *F, double *Es, double *L, int n_eigs, int n_patch, int LANCZOS_ITR)
{
    double r0_norm;

    double *p;
    double *alpha, *beta;
    double *q;
    int i;

    double *T;

	double *lambda; // eigenvalues 

    // generate random r0 with norm 1.
    srand((unsigned int)time(NULL));
    p = (double *)malloc(n_patch * sizeof(double));
    for (i = 0; i < n_patch; i++)
        p[i] = rand();
    r0_norm = norm2(p, n_patch);
    for (i = 0; i < n_patch; i++)
        p[i] /= r0_norm;

    alpha = (double *)malloc( (LANCZOS_ITR + 1) * sizeof(double) );
    beta = (double *)malloc( (LANCZOS_ITR + 1) * sizeof(double) );
    beta[0] = 1.0;
    q = (double *)malloc( n_patch * (LANCZOS_ITR + 1) * sizeof(double) ); 
    memset(q, 0, n_patch * sizeof(double));

    for (i = 1; i <= LANCZOS_ITR; i++) {
        //printf("\nLanczos iteration %d\n", i);
        // Q(:, i) = p / beta(i - 1)
        //printf("beta[%d] = %.9lf\n", i - 1, beta[i - 1]);
        divide_copy(&q[i * n_patch], p, n_patch, &beta[i - 1]);
        // p = Q(:, i - 1)
        cblas_dcopy(n_patch, &q[(i - 1) * n_patch], 1, p, 1);
        // p = L * Q(:, i) - beta(i - 1) * p
        //printf("norm2(p) = %.9lf\n", norm2(p, n_patch));
        cblas_dsymv(CblasColMajor, CblasLower, n_patch, 1.0,  L,
                    n_patch, &q[i * n_patch], 1, -beta[i - 1], p, 1);
        //printf("norm2(p) = %.9lf\n", norm2(p, n_patch));
        // alpha(i) = Q(:, i)' * p
        alpha[i] = cblas_ddot(n_patch, &q[i * n_patch], 1, p, 1);
        //printf("alpha[%d] = %.9lf\n", i, alpha[i]);
        // p = p - alpha(i) * Q(:, i)
        cblas_daxpy(n_patch, -alpha[i], &q[i * n_patch], 1, p, 1);
        // beta(i) = norm(p, 2)
        beta[i] = cblas_dnrm2(n_patch, p, 1);
        //printf("beta[%d] = %.9lf\n", i, beta[i]);
    }

    // build T whose eigensystem approximates that of L.
    // T = diag(alpha) + diag(beta(1:end-1), 1) + diag(beta(1:end-1), -1)
    T = (double *)malloc(LANCZOS_ITR * LANCZOS_ITR * sizeof(double));
    memset(T, 0, LANCZOS_ITR * LANCZOS_ITR * sizeof(double));
    build_tridiagonal(T, alpha, beta, LANCZOS_ITR);

    // compute approximate eigensystem
	lambda = (double *)malloc(LANCZOS_ITR*sizeof(double));
    LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'L', LANCZOS_ITR, T, LANCZOS_ITR,
        lambda);
    // copy specified number of eigenvalues
    memcpy(Es, lambda, n_eigs * sizeof(double));

    // V = Q(:, 1:k) * U
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n_patch, LANCZOS_ITR,
        LANCZOS_ITR, 1.0, &q[n_patch], n_patch, T, LANCZOS_ITR, 0.0, L, n_patch);
    // copy the corresponding eigenvectors
    memcpy(F, L, n_patch * n_eigs * sizeof(double));

    free(lambda);
    free(p);
    free(alpha);
    free(beta);
    free(q);
}

static double norm2(double *v, int length)
{
    int i;
    double sum = 0.0;

    for (i = 0; i < length; i++)
        sum += v[i] * v[i];

    return sqrt(sum);
}

static void divide_copy(double *dest, const double *src, int length, const double *divisor)
{
    double factor = 1.0 / *divisor;
    int i;
    for (i = 0; i < length; i++)
        dest[i] = src[i] * factor;
}

static void build_tridiagonal(double *T, const double *alpha, const double *beta, int Tdim)
{
    int i;
    for (i = 0; i < Tdim; i++) {
        if (i > 0)
            T[(i - 1) + i * Tdim] = beta[i + 1];
        if (i < Tdim - 1)
            T[(i + 1) + i * Tdim] = beta[i + 1];
        T[i + i * Tdim] = alpha[i + 1];
    }
}
