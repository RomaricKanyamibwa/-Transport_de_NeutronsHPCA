/*
 * Université Pierre et Marie Curie
 * Calcul de transport de neutrons
 * Version séquentielle
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#define OUTPUT_FILE "/tmp/romhar/absorbed.dat"
#define THREAD_PER_BLOCK 512 
#define NB_BLOCKS 256

char info[] = "\
Usage:\n\
    neutron-seq H Nb C_c C_s\n\
\n\
    H  : épaisseur de la plaque\n\
    Nb : nombre d'échantillons\n\
    C_c: composante absorbante\n\
    C_s: componente diffusante\n\
\n\
Exemple d'execution : \n\
    neutron-seq 1.0 500000000 0.5 0.5\n\
";

/*
 * générateur uniforme de nombres aléatoires dans l'intervalle [0,1)
 */
struct drand48_data alea_buffer;

void init_uniform_random_number() {
  srand48_r(0, &alea_buffer);
}

float uniform_random_number() {
  double res = 0.0; 
  drand48_r(&alea_buffer,&res);
  return res;
}

/*
 * notre gettimeofday()
 */
double my_gettimeofday(){
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}

__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x*gridDim.x;
    curand_init(16453, id, 0, &state[id]);
}

__global__ void neutron_calculus(curandState *state, float c, float c_c, float h, float* absorbed, int* result, int n, int* c_abs){
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    int pos_ecrit;
    int pos_Thread = id;
    int rt = 0, bt = 0, tt = 0;
    __shared__ int r[THREAD_PER_BLOCK];
    __shared__ int b[THREAD_PER_BLOCK];
    __shared__ int t[THREAD_PER_BLOCK];
    r[threadIdx.x] = 0;
    t[threadIdx.x] = 0;
    b[threadIdx.x] = 0;
    float L;
    float u;
    float d;
    float x;
    while(pos_Thread < n) {
	      d = 0.0;
              x = 0.0;
              while (1) {
	      //result[2]=5;
	      u = curand_uniform (&state[id]);
	      L = -(1 / c) * log(u);
	      x = x + L * cos(d);
	      if (x < 0) {
		r[threadIdx.x] = r[threadIdx.x]+1;
		break;
	      } else if (x >= h) {
		t[threadIdx.x] = t[threadIdx.x]+1;
		break;
	      } else if ((u = curand_uniform (&state[id])) < c_c / c) {
		
		b[threadIdx.x] = b[threadIdx.x]+1;
		pos_ecrit = atomicAdd(c_abs, 1);
		absorbed[pos_ecrit] = x;
		
		break;
	      } else {
		u = curand_uniform (&state[id]);
		d = u * M_PI;
	      }
	    }
	pos_Thread = pos_Thread + gridDim.x*blockDim.x;
	}
	__syncthreads();
	int j = blockDim.x/2;
	while(j>0){
		if(threadIdx.x<j){
			r[threadIdx.x] += r[threadIdx.x + j];
			t[threadIdx.x] += t[threadIdx.x + j];
			b[threadIdx.x] += b[threadIdx.x + j];
		}
		j/=2;
		__syncthreads();
	}
	if(threadIdx.x==0){
		atomicAdd(result,r[0]);
		atomicAdd(result+1,t[0]);
		atomicAdd(result+2,b[0]);
	}
	/*
	if(threadIdx.x==0){
		for(int i=0;i<blockDim.x;i++){
			rt = rt + r[i];
			tt = tt + t[i];
			bt = bt + b[i]; 
		}
		atomicAdd(result,rt);
		atomicAdd(result+1,tt);
		atomicAdd(result+2,bt);
	}
	*/
}

/*
 * main()
 */

int main(int argc, char *argv[]) {
  // La distance moyenne entre les interactions neutron/atome est 1/c. 
  // c_c et c_s sont les composantes absorbantes et diffusantes de c. 
  float c, c_c, c_s;
  // épaisseur de la plaque
  float h;
  // distance parcourue par le neutron avant la collision
  float L;
  // direction du neutron (0 <= d <= PI)
  float d;
  // variable aléatoire uniforme
  float u;
  // position de la particule (0 <= x <= h)
  float x;
  // nombre d'échantillons
  int n;
  // nombre de neutrons refléchis, absorbés et transmis
  int* result = (int *) calloc(3, sizeof(int)); //r, t, b
  // chronometrage
  double start, finish;
  int i, j = 0; // compteurs 

  //perf files
  FILE *perf = fopen("../perform.txt", "a+");
  FILE *perf_gnuplot = fopen("../perform_gnuplot.txt", "a+");
  char str[512];
  //char tmp[128];

  if( argc == 1)
    fprintf( stderr, "%s\n", info);

  // valeurs par defaut
  h = 1.0;
  n = 500000000;
  c_c = 0.5;
  c_s = 0.5;

  // recuperation des parametres
  if (argc > 1)
    h = atof(argv[1]);
  if (argc > 2)
    n = atoi(argv[2]);
  if (argc > 3)
    c_c = atof(argv[3]);
  if (argc > 4)
    c_s = atof(argv[4]);
  c = c_c + c_s;

  // affichage des parametres pour verificatrion
  printf("Épaisseur de la plaque : %4.g\n", h);
  printf("Nombre d'échantillons  : %d\n", n);
  printf("C_c : %g\n", c_c);
  printf("C_s : %g\n", c_s);
  printf("Thread per block : %d\n",THREAD_PER_BLOCK);

  float *absorbed;
  absorbed = (float *) calloc(n, sizeof(float));
  int nb_thread = THREAD_PER_BLOCK;
  dim3 threadsParBloc(nb_thread,1,1);
  dim3 nbBlocks(NB_BLOCKS,1,1);
  float* absorbed_gpu;
  int* result_gpu;
  int* c_abs;
  curandState* d_state;
  cudaMalloc(&d_state, nb_thread*nbBlocks.x*sizeof(curandState));
  cudaMalloc(&absorbed_gpu, n*sizeof(float));
  cudaMemset(absorbed_gpu,0.0,n*sizeof(float));
  cudaMalloc(&result_gpu, 3*sizeof(int));
  cudaMalloc(&c_abs, sizeof(int));
  cudaMemset(c_abs,0,sizeof(int));
  start = my_gettimeofday();
  cudaMemcpy(result_gpu, result, 3*sizeof(int),cudaMemcpyHostToDevice);
  setup_kernel<<<nbBlocks, threadsParBloc >>>(d_state);
  neutron_calculus<<<nbBlocks, threadsParBloc >>>(d_state, c, c_c, h, absorbed_gpu, result_gpu, n, c_abs);
  cudaMemcpy(absorbed, absorbed_gpu, n*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(result, result_gpu, 3*sizeof(int),cudaMemcpyDeviceToHost);
  finish = my_gettimeofday();
  printf("\nTemps total de calcul: %.8g sec\n", finish - start);
  sprintf(str,"***************Cuda N:%d ***************\n\
Nb_thread:%d , Nb_Blocs:%d \n\
#Temps total de calcul : %.8g seconde(s)\n\n"
            ,n,THREAD_PER_BLOCK,NB_BLOCKS,finish-start);

fwrite(str,sizeof(char),strlen(str),perf);
sprintf(str,"%d %.8g %d %d \n",n,finish-start,THREAD_PER_BLOCK,NB_BLOCKS);
fwrite(str,sizeof(char),strlen(str),perf_gnuplot);
  /*
  // debut du chronometrage
  start = my_gettimeofday();
  
  init_uniform_random_number();
  for (i = 0; i < n; i++) {
    d = 0.0;
    x = 0.0;

    while (1) {

      u = uniform_random_number();
      L = -(1 / c) * log(u);
      x = x + L * cos(d);
      if (x < 0) {
	r++;
	break;
      } else if (x >= h) {
	t++;
	break;
      } else if ((u = uniform_random_number()) < c_c / c) {
	b++;
	absorbed[j++] = x;
	break;
      } else {
	u = uniform_random_number();
	d = u * M_PI;
      }
    }
  }

  // fin du chronometrage
  finish = my_gettimeofday();
  */
  int r = result[0];
  int t = result[1];
  int b = result[2];
  printf("\nPourcentage des neutrons refléchis : %4.2g\n", (float) r / (float) n);
  printf("Pourcentage des neutrons absorbés : %4.2g\n", (float) b / (float) n);
  printf("Pourcentage des neutrons transmis : %4.2g\n", (float) t / (float) n);
  printf("Nombre de neutrons traites : %d\n", r+b+t);
  printf("\nTemps total de calcul: %.8g sec\n", finish - start);
  printf("Millions de neutrons /s: %.2g\n", (double) n / ((finish - start)*1e6));

  // ouverture du fichier pour ecrire les positions des neutrons absorbés
  FILE *f_handle = fopen(OUTPUT_FILE, "w+");
  if (!f_handle) {
    fprintf(stderr, "Cannot open " OUTPUT_FILE "\n");
    exit(EXIT_FAILURE);
  }

  for (j = 0; j < b; j++)
    fprintf(f_handle, "%f\n", absorbed[j]);

  // fermeture du fichier
  fclose(f_handle);
  printf("Result written in " OUTPUT_FILE "\n\n "); 

  free(absorbed);
  fclose(perf);
  fclose(perf_gnuplot);
  fclose(f_handle);

  return EXIT_SUCCESS;
}

