all: eigenmap_omp
	mv eigenmap_omp ..
noorth: eigenmap_omp_noorth
	mv eigenmap_omp_noorth ..

MATIO=$(HOME)/lib/libmatio.a
LAPACKE=/usr/local/plasma2.5.0/lib/liblapacke.a
ATLAS_PATH=/usr/local/atlas3.10.1/lib/
DEBUG=

eigenmap_omp: eigenmap.o pairweight.o laplacian.o eigs.o lanczos.o
	gcc $(DEBUG) -o $@ $^ $(MATIO) -static $(LAPACKE) \
	-L$(ATLAS_PATH) -llapack -lf77blas -lcblas -latlas \
	-lgfortran -lz -lm -fopenmp -I. -Wl,-upthread_mutex_destroy

eigenmap_omp_noorth: eigenmap.o pairweight.o laplacian.o eigs.o \
					 lanczos_noorth.o
	gcc $(DEBUG) -o $@ $^ $(MATIO) -static $(LAPACKE) \
	-L$(ATLAS_PATH) -llapack -lptf77blas -lptcblas -latlas \
	-lgfortran -lz -lm -fopenmp -I. -Wl,-upthread_mutex_destroy

eigenmap.o: eigenmap.c
	gcc $(DEBUG) -c $< -o $@ -I$(HOME)/include -fopenmp
pairweight.o: pairweight.c
	gcc $(DEBUG) -c $< -o $@ -I$(HOME)/include -fopenmp
laplacian.o: laplacian.c
	gcc $(DEBUG) -c $< -o $@ -I$(HOME)/include -fopenmp
eigs.o: eigs.c
	gcc $(DEBUG) -c $< -o $@ -I$(HOME)/include -fopenmp
lanczos.o: lanczos.c
	gcc $(DEBUG) -c $< -o $@ -I$(HOME)/include -fopenmp
lanczos_noorth.o: lanczos_noorth.c
	gcc $(DEBUG) -c $< -o $@ -I$(HOME)/include -fopenmp

clean:
	rm -f ../eigenmap_omp ../eigenmap_omp_noorth *.o
