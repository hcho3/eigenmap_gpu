#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "eigenmap.h"

void diff_reduce(double *w, double *feat, double *pos, int feat_dim,
                 int pos_dim, int par0, int par1, int n_patch);

/* pairweight calculates and modifies the weight matrix w (symmetric)
 *
 * w: pointer to allocated space for the symmetric weight matrix
 * feat: list of features vectors
 * pos: list of position vectors
 * n_patch: number of patches
 * feat_dim : dimension of each features vector
 * pos_dim: dimension of each position vector
 * par[2]: parameters
 * option: option (not implemented so far)
 */
void pairweight(double *w, int n_patch, double *feat, double *pos,
                int feat_dim[2], int pos_dim, int par[2], int option)
{
    diff_reduce(w, feat, pos, feat_dim[0] * feat_dim[1], pos_dim,
        par[0], par[1], n_patch); 
}

void diff_reduce(double *w, double *feat, double *pos, int feat_dim,
                 int pos_dim, int par0, int par1, int n_patch)
{
    int i, j, k;
    double feat_dist; // running entry sum of d_ij
    double pos_dist;  // running entry sum of f_ij
    int feat_offi; // offset of x_i
    int feat_offj; // offset of x_j
    int pos_offi;  // offset of p_i
    int pos_offj;  // offset of p_j
    double feat_i, feat_j, pos_i, pos_j;
    // temporary local variables for entry sum calculation

    #pragma omp parallel for shared(w, feat, pos) \
        firstprivate(n_patch, feat_dim, pos_dim, par0, par1) \
        private(feat_dist, pos_dist, feat_offi, feat_offj, \
        pos_offi, pos_offj, feat_i, feat_j, pos_i, pos_j, i, j, k)
    for (i = 0; i < n_patch; i++) {
        for (j = 0; j < n_patch; j++) {
            if (i != j) {
                feat_dist = 0.0;
                pos_dist = 0.0;
                feat_offi = i * feat_dim;
                feat_offj = j * feat_dim;
                pos_offi = i * pos_dim;
                pos_offj = j * pos_dim;

                // get the k-th element of difference vector d_ij
                // and add it to feat_dist
                for (k = 0; k < feat_dim; k++) {
                    feat_i = feat[feat_offi + k];
                    feat_j = feat[feat_offj + k];
                    feat_dist += (feat_i - feat_j) * (feat_i - feat_j);
                }

                // get the k-th element of difference vector f_ij
                // and add it to pos_dist
                for (k = 0; k < pos_dim; k++) {
                    pos_i = pos[pos_offi + k];
                    pos_j = pos[pos_offj + k];
                    pos_dist += (pos_i - pos_j) * (pos_i - pos_j);
                }

                w[i + j * n_patch]
                    = exp( -feat_dist / (feat_dim * par0 * par0))
                       * exp( -pos_dist / (pos_dim * par1 * par1));
            }
        }
    }
}
