all: eigenmap_c
	mv eigenmap_c ..
noorth: eigenmap_c_noorth
	mv eigenmap_c_noorth ..

LAPACK_PATH=/usr/local/plasma2.5.0/lib/liblapacke.a

eigenmap_c: eigenmap.o pairweight.o laplacian.o eigs.o lanczos.o
	gcc -o $@ $^ $(HOME)/lib/libmatio.a $(LAPACK_PATH) -lm -lz -llapack \
		-lblas -I$(HOME)/include -I.
eigenmap_c_noorth: eigenmap.o pairweight.o laplacian.o eigs.o lanczos_noorth.o
	gcc -o $@ $^ $(HOME)/lib/libmatio.a $(LAPACK_PATH) -lm -lz -llapack \
		-lblas -I$(HOME)/include -I.

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
