#define TPB 16
#define BPG 256

void read_mat(const char *filename, double **data_array, double **pos_array, size_t *data_dim, size_t *pos_dim);
void pairweight(double *dev_w, int n_patch, double *data, double *pos, int scale[2], int pos_dim, int par[2], int option);
void laplacian(double *dev_l, double *dev_w, int n_patch);
void eigs(double *F, double *Es, double *dev_l, int n_eigs, int n_patch);
