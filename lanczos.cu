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
__global__ void divide_copy(double *dest, const double * src, int length, const double divisor);

void lanczos(double *F, double *Es, double *dev_l, int n_eigs, int n_patch, int LANCZOS_ITR)
{
	cublasHandle_t handle;

    double *r0;
    double r0_norm;
    double one = 1.0;
    double zero = 0.0;

    double *p;
    double *alpha, *neg_alpha, *beta, *neg_beta;
    double *q;
    int i;

    /* workspace for dstedx */
	magma_int_t info;
	magma_int_t lwork, liwork, ldwork;
	magma_int_t *iwork;
	double *work, *dwork;
    double *eigvec, *dev_eigvec; // eigenvectors

    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

    // generate random r0 with norm 1.
    srand((unsigned int)time(NULL));
    r0 = (double *)malloc(n_patch * sizeof(double));
    for (i = 0; i < n_patch; i++)
        r0[i] = rand();
    r0_norm = norm2(r0, n_patch);
    for (i = 0; i < n_patch; i++)
        r0[i] /= r0_norm;

    HANDLE_ERROR( cudaMalloc((void **)&p, n_patch * sizeof(double)) );
    HANDLE_ERROR( cudaMemcpy(p, r0, n_patch * sizeof(double),
                             cudaMemcpyHostToDevice) );
    alpha = (double *)malloc( (LANCZOS_ITR + 1) * sizeof(double) );
    neg_alpha = (double *)malloc( (LANCZOS_ITR + 1) * sizeof(double) );
    beta = (double *)malloc( (LANCZOS_ITR + 1) * sizeof(double) );
    neg_beta = (double *)malloc( (LANCZOS_ITR + 1) * sizeof(double) );
    HANDLE_ERROR( cudaMalloc((void **)&q,
                             n_patch * (LANCZOS_ITR+1) * sizeof(double)) );
    HANDLE_ERROR( cudaMemset(&q[0], 0, n_patch * sizeof(double)) ); // q0 = 0
    beta[0] = 1.0;
    neg_beta[0] = -1.0;

    for (i = 1; i <= LANCZOS_ITR; i++) {
        // Q(:, i) = p / beta(i - 1)
        divide_copy<<<BPG, TPB>>>(&q[i * n_patch], p, n_patch, beta[i - 1]);
        // p = Q(:, i - 1)
        HANDLE_CUBLAS_ERROR(cublasDcopy(handle, n_patch,
        		&q[(i - 1) * n_patch], 1, p, 1) );
        // p = L * Q(:, i) - beta(i - 1) * p
        HANDLE_CUBLAS_ERROR(cublasDsymv(handle, CUBLAS_FILL_MODE_LOWER,
        		n_patch, &one, dev_l, n_patch, &q[i * n_patch], 1,
        		&neg_beta[i - 1], p, 1) );
        // alpha(i) = Q(:, i)' * p
        HANDLE_CUBLAS_ERROR(cublasDdot(handle, n_patch, &q[i * n_patch],
        		1, p, 1, &alpha[i]) );
        neg_alpha[i] = -alpha[i];
        // p = p - alpha(i) * Q(:, i)
        HANDLE_CUBLAS_ERROR( cublasDaxpy(handle, n_patch, &neg_alpha[i],
                &q[i * n_patch], 1, p, 1) );
        // beta(i) = norm(p, 2)
        HANDLE_CUBLAS_ERROR( cublasDnrm2(handle, n_patch, p, 1, &beta[i]) );
        neg_beta[i] = -beta[i];
        cudaDeviceSynchronize();
    }

    lwork = 1 + 4 * LANCZOS_ITR + LANCZOS_ITR * LANCZOS_ITR;
    work = (double *)malloc(lwork * sizeof(double));
    liwork = 3 + 5 * LANCZOS_ITR;
    iwork = (magma_int_t *)malloc(liwork * sizeof(double));
    ldwork = 3 * LANCZOS_ITR * LANCZOS_ITR / 2 + 3 * LANCZOS_ITR;
    HANDLE_ERROR( cudaMalloc(&dwork, ldwork * sizeof(double)) );

    eigvec = (double *)malloc(LANCZOS_ITR * LANCZOS_ITR * sizeof(double));
    HANDLE_ERROR(cudaMalloc(&dev_eigvec, LANCZOS_ITR * LANCZOS_ITR * sizeof(double)));

    // compute approximate eigensystem
    magma_dstedx('I', LANCZOS_ITR, 0, 1, 1, LANCZOS_ITR, &alpha[1],
                 &beta[1], eigvec, LANCZOS_ITR, work, lwork, iwork,
                 liwork, dwork, &info);

	/* Copy specified number of eigenvalues */
	memcpy(Es, &alpha[1], n_eigs * sizeof(double));

    // V = Q(:, 1:k) * U
    HANDLE_ERROR(cudaMemcpy(dev_eigvec, eigvec, LANCZOS_ITR * LANCZOS_ITR * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_CUBLAS_ERROR( cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n_patch, LANCZOS_ITR, LANCZOS_ITR, &one, &q[n_patch],
                n_patch, dev_eigvec, LANCZOS_ITR, &zero, dev_l, n_patch) );
	/* Copy the corresponding eigenvectors */
	HANDLE_ERROR(cudaMemcpy(F, dev_l, n_patch * n_eigs * sizeof(double),
                 cudaMemcpyDeviceToHost) );

    free(eigvec);
	free(iwork);
	free(work);
    free(r0);
    free(alpha);
    free(neg_alpha);
    free(beta);
    free(neg_beta);
    cudaFree(p);
    cudaFree(q);
    cudaFree(dwork);
    cudaFree(dev_eigvec);
    cublasDestroy(handle);
}

static double norm2(double *v, int length)
{
    int i;
    double sum = 0.0;

    for (i = 0; i < length; i++)
        sum += v[i] * v[i];

    return sqrt(sum);
}

__global__ void divide_copy(double *dest, const double * src, int length, const double divisor)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    double factor = 1.0 / divisor;
    while (tid < length) {
        dest[tid] = src[tid] * factor;
        tid += blockDim.x * gridDim.x;
    }
}
