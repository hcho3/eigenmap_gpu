all: eigenmap_c
	mv eigenmap_c ..

eigenmap_c: eigenmap.c pairweight.c laplacian.c eigs.c
	gcc -o eigenmap_c pairweight.c laplacian.c eigs.c eigenmap.c $(HOME)/lib/libmatio.a /usr/local/plasma2.5.0/lib/liblapacke.a -lm -lz -llapack -lblas -I$(HOME)/include -I.

clean:
	rm -f eigenmap_c
