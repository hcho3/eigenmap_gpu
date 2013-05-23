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
__global__ void divide_copy(double *dest, const double * src, int length, const double *divisor);
__global__ void negate_copy(double *dest, const double * src, int length);
__global__ void build_tridiagonal(double *T, const double *alpha, const double *beta, int Tdim);

void lanczos(double *F, double *Es, double *dev_l, int n_eigs, int n_patch, int LANCZOS_ITR)
{
	cublasHandle_t handle;

    double *r0;
    double r0_norm;
    double beta0 = 1.0;
    double neg_beta0 = -1.0;
    double host_one = 1.0;
    double *one;
    double host_zero = 0.0;
    double *zero;

    double *p;
    double *alpha, *neg_alpha, *beta, *neg_beta;
    double *q;
    int i;

    double *T;
    /* workspace for dsyevd */
	magma_int_t info;
	magma_int_t nb, lwork, liwork, ldwa;
	magma_int_t *iwork;
	double *work, *wa;
	double *lambda; // eigenvalues 

    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

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
    HANDLE_ERROR( cudaMalloc((void **)&alpha,(LANCZOS_ITR+1) * sizeof(double)) );
    HANDLE_ERROR( cudaMalloc((void **)&neg_alpha,
                             (LANCZOS_ITR+1) * sizeof(double)) );
    HANDLE_ERROR( cudaMalloc((void **)&beta, (LANCZOS_ITR+1) * sizeof(double)) );
    HANDLE_ERROR( cudaMalloc((void **)&neg_beta,
                             (LANCZOS_ITR+1) * sizeof(double)) );
    HANDLE_ERROR( cudaMalloc((void **)&q,
                             n_patch * (LANCZOS_ITR+1) * sizeof(double)) );
    HANDLE_ERROR( cudaMemset(&q[0], 0, n_patch * sizeof(double)) ); // q0 = 0
    HANDLE_ERROR( cudaMemcpy(&beta[0], &beta0, sizeof(double),
                             cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(&neg_beta[0], &neg_beta0, sizeof(double),
                             cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMalloc((void **)&one, sizeof(double)) );
    HANDLE_ERROR( cudaMalloc((void **)&zero, sizeof(double)) );
    HANDLE_ERROR( cudaMemcpy(one, &host_one, sizeof(double),
                             cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(zero, &host_zero, sizeof(double),
                             cudaMemcpyHostToDevice) );

    for (i = 1; i <= LANCZOS_ITR; i++) {
        // Q(:, i) = p / beta(i - 1)
        divide_copy<<<TPB, BPG>>>(&q[i * n_patch], p, n_patch, &beta[i - 1]);
        // p = Q(:, i - 1)
        HANDLE_CUBLAS_ERROR(cublasDcopy(handle, n_patch,
        		&q[(i - 1) * n_patch], 1, p, 1) );
        // p = L * Q(:, i) - beta(i - 1) * p
        HANDLE_CUBLAS_ERROR(cublasDsymv(handle, CUBLAS_FILL_MODE_LOWER,
        		n_patch, one, dev_l, n_patch, &q[i * n_patch], 1,
        		&neg_beta[i - 1], p, 1) );
        // alpha(i) = Q(:, i)' * p
        HANDLE_CUBLAS_ERROR(cublasDdot(handle, n_patch, &q[i * n_patch],
        		1, p, 1, &alpha[i]) );
        negate_copy<<<1, 1>>>(&neg_alpha[i], &alpha[i], sizeof(double));
        // p = p - alpha(i) * Q(:, i)
        HANDLE_CUBLAS_ERROR( cublasDaxpy(handle, n_patch, &neg_alpha[i],
        		&q[i * n_patch], 1, p, 1) );
        // beta(i) = norm(p, 2)
        HANDLE_CUBLAS_ERROR( cublasDnrm2(handle, n_patch, p, 1, &beta[i]) );
        negate_copy<<<1, 1>>>(&neg_beta[i], &beta[i], sizeof(double));
        cudaDeviceSynchronize();
    }

    // build T whose eigensystem approximates that of L.
    // T = diag(alpha) + diag(beta(1:end-1), 1) + diag(beta(1:end-1), -1)
    HANDLE_ERROR( cudaMalloc((void **)&T,
                  LANCZOS_ITR * LANCZOS_ITR * sizeof(double)) );
    HANDLE_ERROR( cudaMemset(T, 0, LANCZOS_ITR * LANCZOS_ITR * sizeof(double)) );
    build_tridiagonal<<<2, LANCZOS_ITR/2>>>(T, alpha, beta, LANCZOS_ITR);

    // compute approximate eigensystem
	nb = magma_get_dsytrd_nb(LANCZOS_ITR);
	lwork = LANCZOS_ITR * nb + 6 * LANCZOS_ITR + 2 * LANCZOS_ITR * LANCZOS_ITR;
	liwork = 3 + 5 * LANCZOS_ITR;
    ldwa = LANCZOS_ITR;
	lambda = (double *)malloc(LANCZOS_ITR*sizeof(double));
	wa = (double *)malloc(LANCZOS_ITR * LANCZOS_ITR * sizeof(double));
	iwork = (magma_int_t *)malloc(liwork * sizeof(magma_int_t));
	work = (double *)malloc(lwork * sizeof(double));
    magma_dsyevd_gpu('V', 'L', LANCZOS_ITR, T, LANCZOS_ITR, lambda, wa, ldwa,
                     work, lwork, iwork, liwork, &info);
	/* Copy specified number of eigenvalues */
	memcpy(Es, lambda, n_eigs * sizeof(double));

    // V = Q(:, 1:k) * U
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_patch, LANCZOS_ITR,
                LANCZOS_ITR, one, &q[n_patch], n_patch, T, LANCZOS_ITR,
                zero, dev_l, n_patch);
	/* Copy the corresponding eigenvectors */
	HANDLE_ERROR(cudaMemcpy(F, dev_l, n_patch * n_eigs * sizeof(double),
                 cudaMemcpyDeviceToHost) );

	free(iwork);
	free(work);
	free(wa);
	free(lambda);
    free(r0);
    cudaFree(p);
    cudaFree(alpha);
    cudaFree(neg_alpha);
    cudaFree(beta);
    cudaFree(neg_beta);
    cudaFree(q);
    cudaFree(one);
    cudaFree(zero);
    cudaFree(T);
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

__global__ void divide_copy(double *dest, const double * src, int length, const double *divisor)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    double factor = 1.0 / *divisor;
    while (tid < length) {
        dest[tid] = src[tid] * factor;
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void negate_copy(double *dest, const double * src, int length)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < length) {
        dest[tid] = -src[tid];
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void build_tridiagonal(double *T, const double * alpha, const double *beta, int Tdim)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < Tdim) {
        if (i > 0)
            T[(i - 1) + i * Tdim] = beta[i + 1];
        if (i < Tdim - 1)
            T[(i + 1) + i * Tdim] = beta[i + 1];
        T[i + i * Tdim] = alpha[i + 1];

        i += blockDim.x * gridDim.x;
    }
}
