SRCS = neutron-hybride.cu
EXE_NAME = neutron-hybride
LIB=-lm -L/usr/local/cuda/lib64/ -lcuda -lcudart -lcurand -Xcompiler -fopenmp
NVCC=nvcc
NUM_PROCS = $(shell nproc)

ifeq ($(DEBUGGING), y)
 CFLAGS=-g -O3
else
 CFLAGS=-O3
endif 


all: ${EXE_NAME}

% : %.cu
	$(NVCC) $< -o $@ $(OBJECTS) $(LIB) $(CFLAGS) -g

clean:
	rm -f ${EXE_NAME} *.o *~

exec:all
	export OMP_NUM_THREADS=$(NUM_PROCS)
	./$(EXE_NAME)


