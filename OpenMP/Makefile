SRCS = neutron-openmp.c
EXE_NAME = neutron-openmp

CC = gcc
CFLAGS = -Werror -Wall -O3 -ffast-math -mavx -ftree-vectorize -march=native -funroll-loops -g
LIBS = -lm -fopenmp

all: ${EXE_NAME}

% : %.c
	$(CC) $(CFLAGS) $< -o $@ $(OBJECTS) $(LIBS)

clean:
	rm -f ${EXE_NAME} *.o *~
