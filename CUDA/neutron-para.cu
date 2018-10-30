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
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define OUTPUT_FILE "/tmp/absorbed.dat"

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
    /* Each thread gets same seed, a different sequence 
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

__global__ void neutron_calculus(curandState *state, float c, float c_c, float h, float* absorbed, int* result, int n){
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    int pos_Thread = id;
    int r = 0, b = 0, t = 0;
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
		r = r+1;
		break;
	      } else if (x >= h) {
		b=b+1;
		break;
	      } else if ((u = curand_uniform (&state[id])) < c_c / c) {
		t=t+1;
	
		absorbed[pos_Thread] = x;
		break;
	      } else {
		u = curand_uniform (&state[id]);
		d = u * M_PI;
	      }
	    }
	pos_Thread = pos_Thread + gridDim.x*blockDim.x;
	}
	atomicAdd(result,r);
	atomicAdd(result+1,b);
	atomicAdd(result+2,t);
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
  int* result = (int *) calloc(3, sizeof(int)); //r, b, t
  // chronometrage
  double start, finish;
  int i, j = 0; // compteurs 

  if( argc == 1)
    fprintf( stderr, "%s\n", info);

  // valeurs par defaut
  h = 1.0;
  n = 50000;
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


  float *absorbed;
  absorbed = (float *) calloc(n, sizeof(float));
  int nb_thread = 256;
  dim3 threadsParBloc(nb_thread,1,1);
  dim3 nbBlocks(256,1,1);
  float* absorbed_gpu;
  int* result_gpu;
  curandState* d_state;
  cudaMalloc(&d_state, nb_thread*nbBlocks.x*sizeof(curandState));
  cudaMalloc(&absorbed_gpu, n*sizeof(float));
  cudaMalloc(&result_gpu, 3*sizeof(int));

  cudaMemcpy(result_gpu, result, 3*sizeof(int),cudaMemcpyHostToDevice);

  start = my_gettimeofday();
  setup_kernel<<<nbBlocks, threadsParBloc >>>(d_state);
  neutron_calculus<<<nbBlocks, threadsParBloc >>>(d_state, c, c_c, h, absorbed_gpu, result_gpu, n);
  cudaMemcpy(absorbed, absorbed_gpu, n*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(result, result_gpu, 3*sizeof(int),cudaMemcpyDeviceToHost);
  finish = my_gettimeofday();
  printf("\nTemps total de calcul: %.8g sec\n", finish - start);

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
  int b = result[1];
  int t = result[2];
  printf("\nPourcentage des neutrons refléchis : %4.2g\n", (float) r / (float) n);
  printf("Pourcentage des neutrons absorbés : %4.2g\n", (float) b / (float) n);
  printf("Pourcentage des neutrons transmis : %4.2g\n", (float) t / (float) n);

  printf("\nTemps total de calcul: %.8g sec\n", finish - start);
  printf("Millions de neutrons /s: %.2g\n", (double) n / ((finish - start)*1e6));

  // ouverture du fichier pour ecrire les positions des neutrons absorbés
  FILE *f_handle = fopen(OUTPUT_FILE, "w");
  if (!f_handle) {
    fprintf(stderr, "Cannot open " OUTPUT_FILE "\n");
    exit(EXIT_FAILURE);
  }

  for (j = 0; j < b; j++)
    fprintf(f_handle, "%f\n", absorbed[j]);

  // fermeture du fichier
  fclose(f_handle);
  printf("Result written in " OUTPUT_FILE "\n"); 

  free(absorbed);

  return EXIT_SUCCESS;
}

