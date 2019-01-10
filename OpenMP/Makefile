SRCS = neutron-openmp.c
EXE_NAME = neutron-openmp
NUM_PROCS = $(shell nproc)

CC = gcc
CFLAGS = -Werror -Wall -O3 -g
LIBS = -lm -fopenmp

all: ${EXE_NAME}

% : %.c
	$(CC) $(CFLAGS) $< -o $@ $(OBJECTS) $(LIBS)

clean:
	rm -f ${EXE_NAME} *.o *~

exec:all
	export OMP_NUM_THREADS=$(NUM_PROCS)
	./$(EXE_NAME) 
