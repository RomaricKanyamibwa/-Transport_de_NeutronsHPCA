SRCS = neutron-para.cu
EXE_NAME = neutron-para
LIB=-lm -L/usr/local/cuda/lib64/ -lcuda -lcudart -lcurand
NVCC=nvcc 

ifeq ($(DEBUGGING), y)
 CFLAGS=-g 
else
 CFLAGS=-O3
endif 


all: ${EXE_NAME}

% : %.cu
	$(NVCC) $< -o $@ $(OBJECTS) $(LIB) -g

clean:
	rm -f ${EXE_NAME} *.o *~




