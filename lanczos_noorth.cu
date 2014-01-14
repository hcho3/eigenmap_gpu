#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "book.h"
#include "eigenmap.h"
#include <cuda_runtime.h>
#include <magma.h>
#include <cublas_v2.h>

/*
 * lanczos computes the smallest n_eigs eigenvalues for dev_L and the
 * corresponding eigenvectors using the Lanczos algorithm.
 *
 * F: an array (n_patch by n_eigs) to store the eigenvectors
 * Es: an array (1 by n_eigs) to store the eigenvalues
 * dev_L: an array (n_patch by n_patch) representing the Laplacian matrix
 * n_patch: the dimension of dev_L
 */
static double norm2(double *v, int length);
__global__ void divide_copy(double *dest, const double *src, int length,
                            const double divisor);

void lanczos(double *F, double *Es, double *dev_L, int n_eigs, int n_patch,
             int LANCZOS_ITR)
{
    // declare and allocate necessary variables
	cublasHandle_t handle;
    double one = 1.0, zero = 0.0;

    double *b;
    double b_norm;

    double *z, *w;
    double *alpha, *beta;
    double *neg_alpha, *neg_beta;
    double *q;
    int i;

    /* workspace for dstedx */
	magma_int_t info;
	magma_int_t lwork, liwork, ldwork;
	magma_int_t *iwork;
	double *work, *dwork;
    double *eigvec, *dev_eigvec; // eigenvectors

    cublasCreate(&handle);
    alpha = (double *)malloc( (LANCZOS_ITR + 1) * sizeof(double) );
    beta = (double *)malloc( (LANCZOS_ITR + 1) * sizeof(double) );
    neg_alpha = (double *)malloc( (LANCZOS_ITR + 1) * sizeof(double) );
    neg_beta = (double *)malloc( (LANCZOS_ITR + 1) * sizeof(double) );
    HANDLE_ERROR( cudaMalloc((void **)&q,
                             n_patch * (LANCZOS_ITR + 2) * sizeof(double)) );
    HANDLE_ERROR( cudaMemset(&q[0], 0, n_patch * sizeof(double)) );// q_0 <- 0
    HANDLE_ERROR( cudaMalloc((void **)&z, n_patch * sizeof(double)) );
    HANDLE_ERROR( cudaMalloc((void **)&w, n_patch * sizeof(double)) );
    beta[0] = neg_beta[0] = 0.0;

    // make cuBLAS read scalar values from the host
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

    // initialize q_1 with a random unit vector
    srand((unsigned int)time(NULL));
    b = (double *)malloc(n_patch * sizeof(double));
    for (i = 0; i < n_patch; i++)
        b[i] = rand();
    b_norm = norm2(b, n_patch);
    for (i = 0; i < n_patch; i++)
        b[i] /= b_norm;
    HANDLE_ERROR( cudaMemcpy(&q[n_patch], b, n_patch * sizeof(double),
                             cudaMemcpyHostToDevice) ); // q_1 <- b
    free(b);

    for (i = 1; i <= LANCZOS_ITR; i++) {
        // z = L * Q(:, i)
        cublasDsymv(handle, CUBLAS_FILL_MODE_LOWER, n_patch, &one,
                    dev_L, n_patch, &q[i * n_patch], 1, &zero, z, 1);
        // alpha(i) = Q(:, i)' * z;
        cublasDdot(handle, n_patch, &q[i * n_patch], 1, z, 1, &alpha[i]);
        neg_alpha[i] = -alpha[i];
        // z = z - alpha(i) * Q(:, i)
        cublasDaxpy(handle, n_patch, &neg_alpha[i], &q[i * n_patch], 1, z, 1);
        // z = z - beta(i - 1) * Q(:, i - 1);
        cublasDaxpy(handle, n_patch, &neg_beta[i - 1], &q[(i - 1) * n_patch],
                    1, z, 1);

        // beta(i) = norm(z, 2);
        cublasDnrm2(handle, n_patch, z, 1, &beta[i]);
        neg_beta[i] = -beta[i];
        // Q(:, i + 1) = z / beta(i);
        divide_copy<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>
            (&q[(i + 1) * n_patch], z, n_patch, beta[i]);
    }
    cudaDeviceSynchronize();

    // allocate workspace for magma_dstedx
    lwork = 1 + 4 * LANCZOS_ITR + LANCZOS_ITR * LANCZOS_ITR;
    work = (double *)malloc(lwork * sizeof(double));
    liwork = 3 + 5 * LANCZOS_ITR;
    iwork = (magma_int_t *)malloc(liwork * sizeof(double));
    ldwork = 3 * LANCZOS_ITR * LANCZOS_ITR / 2 + 3 * LANCZOS_ITR;
    HANDLE_ERROR( cudaMalloc(&dwork, ldwork * sizeof(double)) );

    eigvec = (double *)malloc(LANCZOS_ITR * LANCZOS_ITR * sizeof(double));
    HANDLE_ERROR(cudaMalloc(&dev_eigvec, LANCZOS_ITR * LANCZOS_ITR *
                 sizeof(double)) );

    // use divide-and-conquer to approximate eigenvalues
    magma_dstedx('I', LANCZOS_ITR, 0, 1, 1, LANCZOS_ITR, &alpha[1],
                 &beta[1], eigvec, LANCZOS_ITR, work, lwork, iwork,
                 liwork, dwork, &info);

	// Copy specified number of eigenvalues
	memcpy(Es, &alpha[1], n_eigs * sizeof(double));

    // extract eigenvectors of L from Q 
    // V = Q(:, 1:k) * U
    HANDLE_ERROR(cudaMemcpy(dev_eigvec, eigvec, LANCZOS_ITR * LANCZOS_ITR *
                 sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_CUBLAS_ERROR( cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n_patch, LANCZOS_ITR, LANCZOS_ITR, &one, &q[n_patch],
                n_patch, dev_eigvec, LANCZOS_ITR, &zero, dev_L,
                n_patch) );
	// Copy the corresponding eigenvectors
	HANDLE_ERROR(cudaMemcpy(F, dev_L, n_patch * n_eigs * sizeof(double),
                 cudaMemcpyDeviceToHost) );

    // clean up
    cublasDestroy(handle);
    free(alpha);
    free(beta);
    free(neg_alpha);
    free(neg_beta);
    cudaFree(q);
    cudaFree(z);
    cudaFree(w);
    free(work);
    free(iwork);
    cudaFree(dwork);
    free(eigvec);
    cudaFree(dev_eigvec);
}

static double norm2(double *v, int length)
{
    int i;
    double sum = 0.0;

    for (i = 0; i < length; i++)
        sum += v[i] * v[i];

    return sqrt(sum);
}

/* divide the src vector by a given divisor and save the
   result to the dest vector */
__global__ void divide_copy(double *dest, const double *src, int length,
                            const double divisor)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    double factor = 1.0 / divisor;
    while (tid < length) {
        dest[tid] = src[tid] * factor;
        tid += blockDim.x * gridDim.x;
    }
}

