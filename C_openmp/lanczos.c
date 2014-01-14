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
 * lanczos computes the smallest n_eigs eigenvalues for L and the
 * corresponding eigenvectors using the Lanczos algorithm.
 *
 * F: an array (n_patch by n_eigs) to store the eigenvectors
 * Es: an array (1 by n_eigs) to store the eigenvalues
 * L: an array (n_patch by n_patch) representing the Laplacian matrix
 * n_patch: the dimension of L
 */
static double norm2(double *v, int length);
static void divide_copy(double *dest, const double *src, int length,
                        const double divisor);

void lanczos(double *F, double *Es, double *L, int n_eigs, int n_patch,
             int LANCZOS_ITR)
{
    double *b;
    double b_norm;

    double *z;
    double *alpha, *beta;
    double *q;
    int i;
    
	double *eigvec; // eigenvectors 

    // generate random b with norm 1.
    srand((unsigned int)time(NULL));
    b = (double *)malloc(n_patch * sizeof(double));
    for (i = 0; i < n_patch; i++)
        b[i] = rand();
    b_norm = norm2(b, n_patch);
    for (i = 0; i < n_patch; i++)
        b[i] /= b_norm;

    alpha = (double *)malloc( (LANCZOS_ITR + 1) * sizeof(double) );
    beta = (double *)malloc( (LANCZOS_ITR + 1) * sizeof(double) );
    beta[0] = 0.0; // beta_0 <- 0
    z = (double *)malloc( n_patch * sizeof(double));
    q = (double *)malloc( n_patch * (LANCZOS_ITR + 2) * sizeof(double) ); 
    memset(&q[0], 0, n_patch * sizeof(double)); // q_0 <- 0
    memcpy(&q[n_patch], b, n_patch * sizeof(double)); // q_1 <- b

    for (i = 1; i <= LANCZOS_ITR; i++) {
        // z = L * Q(:, i)
        cblas_dsymv(CblasColMajor, CblasLower, n_patch, 1.0, L,
                    n_patch, &q[i * n_patch], 1, 0.0, z, 1);
        // alpha(i) = Q(:, i)' * z;
        alpha[i] = cblas_ddot(n_patch, &q[i * n_patch], 1, z, 1);
        // z = z - alpha(i) * Q(:, i)
        cblas_daxpy(n_patch, -alpha[i], &q[i * n_patch], 1, z, 1);
        // z = z - beta(i - 1) * Q(:, i - 1);
        cblas_daxpy(n_patch, -beta[i - 1], &q[(i - 1) * n_patch], 1, z, 1);
        /* re-orthogonalize twice */
        // b = Q(:, 1:i-1)' * z
        cblas_dgemv(CblasColMajor, CblasTrans, n_patch, i - 1, 1.0,
                    &q[n_patch], n_patch, z, 1, 0.0, b, 1);
        // z = Q(:, 1:i-1) * b + (-1) * z
        cblas_dgemv(CblasColMajor, CblasNoTrans, n_patch, i - 1, 1.0,
                    &q[n_patch], n_patch, b, 1, -1.0, z, 1);
        // b = Q(:, 1:i-1)' * z
        cblas_dgemv(CblasColMajor, CblasTrans, n_patch, i - 1, 1.0,
                    &q[n_patch], n_patch, z, 1, 0.0, b, 1);
        // z = Q(:, 1:i-1) * b + (-1) * z
        cblas_dgemv(CblasColMajor, CblasNoTrans, n_patch, i - 1, 1.0,
                    &q[n_patch], n_patch, b, 1, -1.0, z, 1);

        // beta(i) = norm(z, 2);
        beta[i] = cblas_dnrm2(n_patch, z, 1);
        // Q(:, i + 1) = z / beta(i);
        divide_copy(&q[(i + 1) * n_patch], z, n_patch, beta[i]);
    }

    // compute approximate eigensystem
    eigvec = (double *)malloc(LANCZOS_ITR * LANCZOS_ITR * sizeof(double));
    LAPACKE_dstedc(LAPACK_COL_MAJOR, 'I', LANCZOS_ITR, &alpha[1], &beta[1],
                   eigvec, LANCZOS_ITR); 
    // copy specified number of eigenvalues
    memcpy(Es, &alpha[1], n_eigs * sizeof(double));

    // V = Q(:, 1:k) * U
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n_patch,
                LANCZOS_ITR, LANCZOS_ITR, 1.0, &q[n_patch], n_patch, eigvec,
                LANCZOS_ITR, 0.0, L, n_patch);
    // copy the corresponding eigenvectors
    memcpy(F, L, n_patch * n_eigs * sizeof(double));

    free(b);
    free(z);
    free(alpha);
    free(beta);
    free(q);
    free(eigvec);
}

static double norm2(double *v, int length)
{
    int i;
    double sum = 0.0;

    for (i = 0; i < length; i++)
        sum += v[i] * v[i];

    return sqrt(sum);
}

static void divide_copy(double *dest, const double *src, int length,
                        const double divisor)
{
    double factor = 1.0 / divisor;
    int i;
    for (i = 0; i < length; i++)
        dest[i] = src[i] * factor;
}
