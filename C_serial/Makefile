all: eigenmap_c
	mv eigenmap_c ..
noorth: eigenmap_c_noorth
	mv eigenmap_c_noorth ..

MATIO=$(HOME)/lib/libmatio.a
LAPACKE=/usr/local/plasma2.5.0/lib/liblapacke.a
ATLAS_PATH=/usr/local/atlas3.10.1/lib/

eigenmap_c: eigenmap.o pairweight.o laplacian.o eigs.o lanczos.o
	gcc -o $@ $^ $(MATIO) --static $(LAPACKE) -L$(ATLAS_PATH) \
		-llapack -lcblas -lf77blas -latlas -lgfortran -lz -lm -lpthread -I.
eigenmap_c_noorth: eigenmap.o pairweight.o laplacian.o eigs.o lanczos_noorth.o
	gcc -o $@ $^ $(MATIO) --static $(LAPACKE) -L$(ATLAS_PATH) \
		-llapack -lcblas -lf77blas -latlas -lgfortran -lz -lm -lpthread -I.

eigenmap.o: eigenmap.c
	gcc -c $< -o $@ -I$(HOME)/include
pairweight.o: pairweight.c
	gcc -c $< -o $@ -I$(HOME)/include
laplacian.o: laplacian.c
	gcc -c $< -o $@ -I$(HOME)/include
eigs.o: eigs.c
	gcc -c $< -o $@ -I$(HOME)/include
lanczos.o: lanczos.c
	gcc -c $< -o $@ -I$(HOME)/include
lanczos_noorth.o: lanczos_noorth.c
	gcc -c $< -o $@ -I$(HOME)/include

clean:
	rm -f ../eigenmap_c ../eigenmap_c_noorth *.o
