all: eigenmap_c
	mv eigenmap_c ..

eigenmap_c: eigenmap.o pairweight.o laplacian.o eigs.o lanczos.o
	gcc -o $@ $^ $(HOME)/lib/libmatio.a /usr/local/plasma2.5.0/lib/liblapacke.a -lm -lz -llapack -lblas -I$(HOME)/include -I.

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

clean:
	rm -f ../eigenmap_c *.o
