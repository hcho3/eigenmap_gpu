all: eigenmap legacy
legacy: eigenmap_legacy
noorth: eigenmap_noorth

ARCH=-arch=sm_35

# default number of threads and blocks
TPB=128
BPG=2048
GRID=-D THREADS_PER_BLOCK=$(TPB) -D BLOCKS_PER_GRID=$(BPG)

# SYSTEM PATHS
MAGMA_INC=/usr/local/magma/include
MAGMA_LIB=/usr/local/magma/lib
CUDA_LIB=/usr/local/cuda/lib64
ATLAS_LIB=/usr/local/atlas3.10.1/lib


MATIO_LIB=./matio/lib/libmatio.a
MATIO_INC=./matio/include
MAGMA=$(MAGMA_LIB)/libmagma.a \
	  $(MAGMA_LIB)/libmagmablas.a
ATLAS=-L$(ATLAS_LIB) -llapack -lcblas -lf77blas -latlas -lgfortran -lz -lm
CUDA=-L$(CUDA_LIB) -lcublas -lcudadevrt

eigenmap.o: eigenmap.cu eigenmap.h
	nvcc $(ARCH) -rdc=true -c $< -o $@ -Xcompiler -fPIC -I$(MATIO_INC)
	
eigenmap_legacy.o: eigenmap_legacy.cu eigenmap.h
	nvcc $(ARCH) -rdc=true -c $< -o $@ -Xcompiler -fPIC -I$(MATIO_INC)

pairweight.o: pairweight.cu
	nvcc $(ARCH) -rdc=true -c $< -o $@ -Xcompiler -fPIC -I$(MATIO_INC)
	
laplacian.o: laplacian.cu eigenmap.h
	nvcc $(ARCH) -rdc=true -c $< -o $@ -Xcompiler -fPIC -I$(MATIO_INC) \
		 $(GRID)
	
eigs.o: eigs.cu
	nvcc $(ARCH) -rdc=true -c $< -o $@ -Xcompiler -fPIC -I$(MATIO_INC) \
		 -I$(MAGMA_INC)
	
book.o: book.cu
	nvcc $(ARCH) -rdc=true -c $< -o $@ -Xcompiler -fPIC -I$(MATIO_INC)

lanczos.o: lanczos.cu eigenmap.h
	nvcc $(ARCH) -rdc=true -c $< -o $@ -Xcompiler -fPIC -I$(MATIO_INC) \
		 -I$(MAGMA_INC) $(GRID)

lanczos_noorth.o: lanczos_noorth.cu eigenmap.h
	nvcc $(ARCH) -rdc=true -c $< -o $@ -Xcompiler -fPIC -I$(MATIO_INC) \
		 $(GRID)

eigenmap: eigenmap.o pairweight.o laplacian.o book.o lanczos.o
	nvcc $(ARCH) -rdc=true -o $@ $^ $(MATIO_LIB) $(MAGMA) $(ATLAS) $(CUDA)

eigenmap_legacy: eigenmap_legacy.o pairweight.o laplacian.o book.o eigs.o
	nvcc $(ARCH) -rdc=true -o $@ $^ $(MATIO_LIB) $(MAGMA) $(ATLAS) $(CUDA)

eigenmap_noorth: eigenmap.o pairweight.o laplacian.o book.o lanczos_noorth.o
	nvcc $(ARCH) -rdc=true -o $@ $^ $(MATIO_LIB) $(MAGMA) $(ATLAS) $(CUDA)

clean:
	rm -f eigenmap eigenmap_legacy eigenmap_noorth *.o
