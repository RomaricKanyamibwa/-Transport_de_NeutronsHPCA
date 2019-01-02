SRCS = neutron-seq.c
EXE_NAME = neutron-seq

CC = gcc
CFLAGS = -Wall -Werror -O3 # -std=c11
LIBS = -lm

CUDA=CUDA
OpenMP=OpenMP
Hybride=Hybride

all: ${EXE_NAME}
	cd $(CUDA) && $(MAKE)
	cd $(OpenMP) && $(MAKE)
	cd $(Hybride) && $(MAKE)

% : %.c
	$(CC) $(CFLAGS) $< -o $@ $(OBJECTS) $(LIBS)

clean:
	rm -f ${EXE_NAME} *.o *~
	cd $(CUDA) && $(MAKE) clean
	cd $(OpenMP) && $(MAKE) clean
	cd $(Hybride) && $(MAKE) clean
	
exec:all
	sh benchmarking.sh
