#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <curand.h> //bibliothèque pour la generation de nombres aléatoires  
#include <curand_kernel.h>
#include <omp.h>

#define OUTPUT_FILE "/tmp/3302011/absorbed.dat"
#define NB_THREADS_PER_BLOCK 1024
#define NB_BLOCKS 256
#define NbthreadsOmp 32

char info[] = "\
Usage:\n\
    neutron-gpu H Nb C_c C_s\n\
\n\
    H  : épaisseur de la plaque\n\
    Nb : nombre d'échantillons\n\
    C_c: composante absorbante\n\
    C_s: componente diffusante\n\
\n\
Exemple d'execution :\n\
    neutron-gpu 1.0 500000000 0.5 0.5\n\
\n\
";

/*
 * notre gettimeofday()
 */
double my_gettimeofday(){
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec* 1.0e-6L);
}

struct drand48_data alea_buffer;

void init_uniform_random_number(int seed) {
  srand48_r(seed + omp_get_thread_num(), &alea_buffer);
}

float uniform_random_number() {
  double res = 0.0; 
  drand48_r(&alea_buffer,&res);
  return res;
}

__global__ void setup_kernel(curandState* state, unsigned long seed)
{
    int id_thread = threadIdx.x + blockIdx.x*blockDim.x;
    curand_init(seed, id_thread, 0, &state[id_thread]);
}

__device__ float generate(curandState* globalState, int ind) 
{
    curandState localState = globalState[ind];
    float random = curand_uniform(&localState);
    globalState[ind] = localState;
    return random;
}

__global__ void kernel(curandState* globalState, float* absorbed, float h, int n, float c, float c_c, float c_s, int* result) //uniquement les elements qui sont intialisés dans le main + r b et t
{
  float d; 
  float x; 
  float L;
  float u;
  int i = blockDim.x*blockIdx.x + threadIdx.x; //sert de compteur
  int gi = i;
  int prev;
  __shared__ int r_local[NB_THREADS_PER_BLOCK]; 
  __shared__ int t_local[NB_THREADS_PER_BLOCK]; 
  __shared__ int b_local[NB_THREADS_PER_BLOCK];
  r_local[threadIdx.x] = 0; //on initialise à zéro le tableau
  t_local[threadIdx.x] = 0; 
  b_local[threadIdx.x] = 0;  
    
  while(i<n){ //i doit s'incrémenter mais pas gi
  d = 0.0; 
  x = 0.0;
  while (1) { 
    u = generate(globalState,gi); 
    L = -(1 / c) * log(u);
    x = x + L * cos(d);
    if (x < 0) { //reflechi  
    r_local[threadIdx.x] = r_local[threadIdx.x] + 1;
     break;
    } 
    else if (x >= h) { //transmis 
    t_local[threadIdx.x] = t_local[threadIdx.x] + 1;
    break;
    } 
    else if ((u = generate(globalState,gi)) < c_c / c) { //absorbé
     b_local[threadIdx.x] = b_local[threadIdx.x] + 1; 
     prev = atomicAdd(result+3,1); //communication interphread pas possible donc on veut l'atomicAdd pour pas écrire de manière concurente (on donne la main à 1 thread) 
     absorbed[prev] = x;
      break;
    } 
    else {
      u = generate(globalState,gi);
      d = u * M_PI; //direction
    } 
  } //boucle while(1)
  i += (gridDim.x*blockDim.x);
}//while(i<n) //tant qu'on a pas traité tous les neutrons 

  __syncthreads(); //synchronize the local threads writing to the local memory cache 

  int j = blockDim.x / 2; 

  while(j>0)
  {
    if(threadIdx.x < j)
    {
      r_local[threadIdx.x]+=r_local[threadIdx.x+j];
      t_local[threadIdx.x]+=t_local[threadIdx.x+j];
      b_local[threadIdx.x]+=b_local[threadIdx.x+j];
    }
    j/=2; 
    __syncthreads();
  }

if(threadIdx.x == 0){//le premier thread va faire les calculs 
  atomicAdd(result,r_local[0]); 
  atomicAdd(result+1,b_local[0]);
  atomicAdd(result+2,t_local[0]);
}//fin if 

}

int main(int argc, char *argv[]) {

  // chronometrage
  double start, finish;

  if( argc == 1)
    fprintf( stderr, "%s\n", info);

  float c, c_c, c_s;
  float h;
  int r, b, t;
  int n,j; 
  int n1;
  int i; 
  float d, L, u, x;
  dim3 NbBlocks;
  int prev = 0; 
  int r2, b2, t2, j2;

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
  
  c = c_s + c_c; 
  r = b = t = j = 0;
  r2 = b2 = t2 = j2 = 0;
  n1 = n - n/100; 
  
  // affichage des parametres pour verificatrion
  printf("Épaisseur de la plaque : %4.g\n", h);
  printf("Nombre d'échantillons  : %d\n", n);
  printf("C_c : %g\n", c_c);
  printf("C_s : %g\n", c_s);
  float* absorbed_CPU;
  float* absorbed_GPU;
  float* absorbed_CPU2; 
  int* result_CPU; 
  int* result_GPU;
  curandState* devStates;
 
  /* Définition du nombre de threads et de la taille de la grille */
 
  dim3 NbThreadsParBloc(256,1,1);

  NbBlocks.x = NB_BLOCKS;
  
  printf("nombre de threads par bloc : %4.2d\n",NB_THREADS_PER_BLOCK);
  printf("nombre de blocs : %4.2d\n",NbBlocks.x);
  printf("nombre de neutrons traités par le GPU : %d\n", n1);
  int n2;
  n2 = n - n1; 
  printf("nombre de neutrons traités par le CPU : %d\n",n2);
  
  NbBlocks.y = 1;
  NbBlocks.z = 1; 

  /* Allocation de la mémoire */
  absorbed_CPU = (float *) calloc(n,sizeof(float)); //sur CPU 
  absorbed_CPU2 = (float *) calloc(n1,sizeof(float)); //sur CPU

  result_CPU = (int *) calloc(4,sizeof(int)); //sur CPU
  cudaMalloc((void**) &absorbed_GPU, n*sizeof(float)); //sur GPU
  cudaMalloc (&devStates, NbThreadsParBloc.x*NbBlocks.x*sizeof(curandState));
  cudaMalloc((void**) &result_GPU, 4*sizeof(int));

  cudaMemcpy(absorbed_CPU2, absorbed_GPU, n1*sizeof(float), cudaMemcpyHostToDevice); //copie du absorbed CPU dans GPU 
  cudaMemcpy(result_CPU, result_GPU, 4*sizeof(int), cudaMemcpyHostToDevice);

  // debut du chronometrage
  start = my_gettimeofday();
  omp_set_num_threads(NbthreadsOmp);
  printf("thread début : %d\n",omp_get_thread_num());

#pragma omp parallel shared(absorbed_CPU)
{
 init_uniform_random_number(omp_get_num_threads());
 printf("thread GPU : %d\n",omp_get_thread_num()); 

 #pragma omp master
 {
   setup_kernel <<<NbBlocks,NbThreadsParBloc>>> (devStates,unsigned(time(NULL)));  //initialisation de l'état curandState pour chaque thread
   kernel<<<NbBlocks, NbThreadsParBloc>>>(devStates, absorbed_GPU, h, n1, c, c_c, c_s, result_GPU); //génération des positions absorbed pour GPU
  //on renvoie aussi r, t et b pour l'affichage plus loin dans le code 
  cudaMemcpy(absorbed_CPU2, absorbed_GPU, n1*sizeof(float), cudaMemcpyDeviceToHost); //copie du absorbed GPU dans CPU 
  cudaMemcpy(result_CPU, result_GPU, 4*sizeof(int), cudaMemcpyDeviceToHost);
  }

#pragma omp for reduction(+:j2,r2,b2,t2) private(d,x,u,L,i) schedule(static)
 for (i = 0; i <(n-n1); i++) {  
    d = 0.0;
    x = 0.0;

    while (1) {
      u = uniform_random_number();
      L = -(1 / c) * log(u);
      x = x + L * cos(d);
      if (x < 0) {
      r2++;
      break;
      } else if (x >= h) {
      t2++;
      break;
      } else if ((u = uniform_random_number()) < c_c / c) {
      b2++;
      prev = j2; 
      #pragma omp atomic 
      j2++; 
      absorbed_CPU[prev] = x;
      break;
      } else {
      u = uniform_random_number();
      d = u * M_PI;
      }
    }
  }
}//pragma omp parallel

  for(int k = 0; k<n1; k++)
  {
   absorbed_CPU[k] = absorbed_CPU2[k];  
   }
  
  r = result_CPU[0] + r2; 
  b = result_CPU[1] + b2; 
  t = result_CPU[2] + t2; 
  j = result_CPU[3] + j2;


  // fin du chronometrage
  finish = my_gettimeofday();

  printf("\nPourcentage des neutrons refléchis : %4.2g\n", (float) r / (float) n);
  printf("Pourcentage des neutrons absorbés : %4.2g\n", (float) b / (float) n);
  printf("Pourcentage des neutrons transmis : %4.2g\n", (float) t / (float) n);

  printf("\nTemps total de calcul: %.8g sec\n", finish - start);
  printf("Millions de neutrons /s: %.2g\n", (double) n / ((finish - start)*1e6));

  printf("réfléchis = %d, absorbés = %d, transmis = %d\n", r, b,t);
  printf("Total traité: %d\n", r + b +t);

  // ouverture du fichier pour ecrire les positions des neutrons absorbés
  FILE *f_handle = fopen(OUTPUT_FILE, "w");
  if (!f_handle) {
     fprintf(stderr, "Cannot open " OUTPUT_FILE "\n");
     exit(EXIT_FAILURE);
 }
  
  for (j = 0; j < b; j++)
     fprintf(f_handle, "%f\n", absorbed_CPU[j]);
  
  // fermeture du fichier
  fclose(f_handle);
  printf("Result written in " OUTPUT_FILE "\n"); 

  cudaFree(absorbed_GPU); 
  cudaFree(devStates);
  cudaFree(result_GPU);
  free(result_CPU); 
  free(absorbed_CPU);
  free(absorbed_CPU2);

  return EXIT_SUCCESS;
}