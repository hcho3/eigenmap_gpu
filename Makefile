all: eigenmap

ARCH=-arch=sm_35

SOURCES=eigenmap.cu pairweight.cu laplacian.cu eigs.cu book.cu
OBJECTS=$(SOURCES:.cu=.o)

MAGMA_PATH=/usr/local/magma/lib
CUDA_PATH=/usr/local/cuda/lib64
STATIC_LIBS=$(HOME)/lib/libmatio.a \
			$(MAGMA_PATH)/libmagma.a \
			$(MAGMA_PATH)/libmagmablas.a
SHARED_LIBS=-llapack -lblas -lm -lz
CUDA=-L$(CUDA_PATH) -lcublas -lcudadevrt


$(OBJECTS): $(SOURCES)
	nvcc $(ARCH) -rdc=true -c $*.cu -o $@ -Xcompiler -fPIC -I$(HOME)/include

eigenmap: $(OBJECTS)
	nvcc $(ARCH) -rdc=true -o $@ $(OBJECTS) $(STATIC_LIBS) $(SHARED_LIBS) $(CUDA)

clean:
	rm -f eigenmap *.o
