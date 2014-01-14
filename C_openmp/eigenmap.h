void pairweight(double *w, int n_patch, double *feat, double *pos,
                int feat_dim[2], int pos_dim, int par[2], int option);
void laplacian(double *w, int n_patch);
void eigs(double *F, double *Es, double *l, int n_eigs, int n_patch);
void lanczos(double *F, double *Es, double *L, int n_eigs, int n_patch,
             int LANCZOS_ITR);
