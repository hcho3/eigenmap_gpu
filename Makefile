all: eigenmap
legacy: eigenmap_legacy

ARCH=-arch=sm_35

# default number of threads and blocks
TPB=128
BPG=2048

MAGMA_PATH=/usr/local/magma/lib
CUDA_PATH=/usr/local/cuda/lib64
STATIC_LIBS=$(HOME)/lib/libmatio.a \
			$(MAGMA_PATH)/libmagma.a \
			$(MAGMA_PATH)/libmagmablas.a
SHARED_LIBS=-llapack -lblas -lm -lz
CUDA=-L$(CUDA_PATH) -lcublas -lcudadevrt
GRID=-D THREADS_PER_BLOCK=$(TPB) -D BLOCKS_PER_GRID=$(BPG)

eigenmap.o: eigenmap.cu eigenmap.h
	nvcc $(ARCH) -rdc=true -c $< -o $@ -Xcompiler -fPIC -I$(HOME)/include
	
eigenmap_legacy.o: eigenmap_legacy.cu eigenmap.h
	nvcc $(ARCH) -rdc=true -c $< -o $@ -Xcompiler -fPIC -I$(HOME)/include

pairweight.o: pairweight.cu
	nvcc $(ARCH) -rdc=true -c $< -o $@ -Xcompiler -fPIC -I$(HOME)/include
	
laplacian.o: laplacian.cu eigenmap.h
	nvcc $(ARCH) -rdc=true -c $< -o $@ -Xcompiler -fPIC -I$(HOME)/include $(GRID)
	
eigs.o: eigs.cu
	nvcc $(ARCH) -rdc=true -c $< -o $@ -Xcompiler -fPIC -I$(HOME)/include
	
book.o: book.cu
	nvcc $(ARCH) -rdc=true -c $< -o $@ -Xcompiler -fPIC -I$(HOME)/include

lanczos.o: lanczos.cu eigenmap.h
	nvcc $(ARCH) -rdc=true -c $< -o $@ -Xcompiler -fPIC -I$(HOME)/include $(GRID)

eigenmap: eigenmap.o pairweight.o laplacian.o book.o lanczos.o
	nvcc $(ARCH) -rdc=true -o $@ $^ $(STATIC_LIBS) $(SHARED_LIBS) $(CUDA)

eigenmap_legacy: eigenmap_legacy.o pairweight.o laplacian.o book.o eigs.o
	nvcc $(ARCH) -rdc=true -o $@ $^ $(STATIC_LIBS) $(SHARED_LIBS) $(CUDA)

clean:
	rm -f eigenmap eigenmap_legacy *.o
