#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <matio.h>
#include <sys/time.h>
#include "eigenmap.h"

int NUM_EIGS;
char filename[30];

double GetTimerValue(struct timeval time_1, struct timeval time_2);
void read_mat(const char *filename, double **data_array, double **pos_array, size_t *data_dim, size_t *pos_dim);
void write_mat(double *F, double *Es, int n_patch);
int main(int argc, char **argv)
{
	double *data_array, *pos_array;
	size_t data_dim[3] = {0};
	size_t pos_dim[2] = {0};
	double *w;
	double *l;
	double *F, *Es;
	int n_patch;
	int scale[2];
	int par[2];
	struct timeval timer1, timer2;

    int i;

	if (argc != 5) {
		printf("Usage: ./eigenmap_c [MAT file containing patches] "
		       "[# of eigenvalues] [parameter 1] [parameter 2]\n");
		return 0;
	}
	if (sscanf(argv[2], "%d", &NUM_EIGS) < 1 || NUM_EIGS < 1 ||
		sscanf(argv[3], "%d", &par[0]) < 1 || par[0] < 1 ||
		sscanf(argv[4], "%d", &par[1]) < 1 || par[1] < 1) {
		printf("Usage: ./eigenmap_c [MAT file containing patches] "
		       "[# of eigenvalues] [parameter 1] [parameter 2]\n");
		return 0;
	}
    
    // DEBUG
    for (i = 0; argv[1][i] != '.'; i++)
        filename[i] = argv[1][i];
    filename[i] = '\0';

	// Read in the matlab file that contains patches structure.
	read_mat(argv[1], &data_array, &pos_array, data_dim, pos_dim);
	n_patch = (int) data_dim[2];
	scale[0] = (int) data_dim[0];
	scale[1] = (int) data_dim[1];
	printf("%lux%lux%lu\n", data_dim[0], data_dim[1], data_dim[2]);
		// memory allocation
	w = (double *)malloc(n_patch * n_patch * sizeof(double));
	memset(w, 0, n_patch * n_patch * sizeof(double));
	l = (double *)malloc(n_patch * n_patch * sizeof(double));
	memset(l, 0, n_patch * n_patch * sizeof(double));
	F = (double *)malloc(n_patch * NUM_EIGS * sizeof(double));
	Es = (double *)malloc(NUM_EIGS * sizeof(double));
	
	gettimeofday(&timer1, NULL);
	// Compute the weight matrix W. And W = W + W'
	pairweight(w, n_patch, data_array, pos_array, scale, pos_dim[0], par, 1);
	// Compute the Laplacian L
	laplacian(l, w, n_patch);	
	// Compute eigenvalues and eigen vectors of L
	eigs(F, Es, l, NUM_EIGS, n_patch);
	gettimeofday(&timer2, NULL);
	printf("Time to compute: %.3lf ms\n", GetTimerValue(timer1, timer2));

	// output the result
	write_mat(F, Es, n_patch);

	// free memory
	free(w);
	free(l);
	free(F);
	free(Es);
}

void read_mat(const char *filename, double **data_array, double **pos_array, size_t *data_dim, size_t *pos_dim)
{
    mat_t *matfp;
    matvar_t *patches, *data, *pos;

    matfp = Mat_Open(filename, MAT_ACC_RDONLY);
    if(matfp == NULL) {
        fprintf(stderr, "Error opening MAT file \"%s\"!\n", filename);
        exit(EXIT_FAILURE);
    }

    if((patches = Mat_VarReadNext(matfp)) == NULL) {
        fprintf(stderr, "Error reading the variable.\n");
        Mat_Close(matfp);
        exit(EXIT_FAILURE);
    }

    if(patches->data_type != MAT_T_STRUCT) {
        fprintf(stderr, "The variable %s is not a valid structure. Type: %d\n", patches->name, patches->data_type);
        Mat_VarFree(patches);
        Mat_Close(matfp);
        exit(EXIT_FAILURE);
    }

    data = Mat_VarGetStructFieldByName(patches, "data", 0);
    pos = Mat_VarGetStructFieldByName(patches, "pos", 0);

    if(data == NULL || pos == NULL) {
        fprintf(stderr, "The variable %s is not a valid structure.\n", patches->name);
        Mat_VarFree(patches);
        Mat_Close(matfp);
        exit(EXIT_FAILURE);
    }

	// Allocate memory for data_array and pos_array in heap space. Modify them accordingly
	*data_array = (double *)malloc(data->dims[0] * data->dims[1] * data->dims[2] * sizeof(double));
	*pos_array = (double *)malloc(pos->dims[0] * pos->dims[1] * sizeof(double));
	
	memcpy(*data_array, data->data, data->dims[0] * data->dims[1] * data->dims[2] * sizeof(double));
	memcpy(*pos_array, pos->data, pos->dims[0] * pos->dims[1] * sizeof(double));

	// Pass data_dim and pos_dim to main
	memcpy(data_dim, data->dims, 3 * sizeof(size_t));
	memcpy(pos_dim, pos->dims, 2 * sizeof(size_t));

	Mat_VarFree(patches);
	Mat_Close(matfp);
}
void write_l(double *l, int n_patch)
{
    mat_t *matfp;
    matvar_t *lm;
	size_t l_dims[2] = {n_patch, n_patch};

	matfp = Mat_CreateVer("l.mat", NULL, MAT_FT_DEFAULT);
	if(matfp == NULL) {
		fprintf(stderr, "Error creating MAT file \"w.mat\"\n");
		exit(EXIT_FAILURE);
	}

	lm = Mat_VarCreate("l", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, l_dims, l, 0);
	if(lm == NULL) {
		fprintf(stderr, "Error creating MAT variable l.\n");
		Mat_Close(matfp);
		exit(EXIT_FAILURE);
	} else {
		Mat_VarWrite(matfp, lm, MAT_COMPRESSION_NONE);
		Mat_VarFree(lm);
	}
	
	Mat_Close(matfp);

}
void write_mat(double *F, double *Es, int n_patch)
{
    mat_t *matfp;
    matvar_t *Fm, *Esm;
	size_t F_dims[2] = {n_patch, NUM_EIGS};
	size_t Es_dims[2] = {NUM_EIGS, 1};

	matfp = Mat_CreateVer("F.mat", NULL, MAT_FT_DEFAULT);
	if(matfp == NULL) {
		fprintf(stderr, "Error creating MAT file \"F.mat\"\n");
		exit(EXIT_FAILURE);
	}

	Fm = Mat_VarCreate("F", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, F_dims, F, 0);
	if(Fm == NULL) {
		fprintf(stderr, "Error creating MAT variable F.\n");
		Mat_Close(matfp);
		exit(EXIT_FAILURE);
	} else {
		Mat_VarWrite(matfp, Fm, MAT_COMPRESSION_NONE);
		Mat_VarFree(Fm);
	}
	
	Mat_Close(matfp);

	matfp = Mat_CreateVer("Es.mat", NULL, MAT_FT_DEFAULT);
	
	if(matfp == NULL) {
		fprintf(stderr, "Error creating MAT file \"Es.mat\"\n");
		Mat_Close(matfp);
		exit(EXIT_FAILURE);
	}

	Esm = Mat_VarCreate("Es", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, Es_dims, Es, 0);
	if(Esm == NULL) {
		fprintf(stderr, "Error creating MAT variable Es.\n");
		Mat_Close(matfp);
		exit(EXIT_FAILURE);
	} else {
		Mat_VarWrite(matfp, Esm, MAT_COMPRESSION_NONE);
		Mat_VarFree(Esm);
	}

	Mat_Close(matfp);
}

double GetTimerValue(struct timeval time_1, struct timeval time_2)
{
  int sec, usec;
  sec  = time_2.tv_sec  - time_1.tv_sec;
  usec = time_2.tv_usec - time_1.tv_usec;
  return (1000.*(double)(sec) + (double)(usec) * 0.001);
}