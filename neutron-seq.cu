#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <curand.h> //bibliothèque pour la generation de nombres aléatoires  
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

static int iDivUp(int a, int b){ //donne la division de a par b
  return ((a % b != 0) ? (a / b + 1) : (a / b));
}


/*
 * notre gettimeofday()
 */
double my_gettimeofday(){
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec* 1.0e-6L);
}

/*
  Fonction qui va servir à initialiser des champs pour des nombres aléatoires avec des graines seed. Chaque thread aura un nombre aléatoire différent. 
*/

__global__ void setup_kernel(curandState* state, unsigned long seed)
{
    int id_thread = threadIdx.x + blockIdx.x*blockDim.x;
    //threadIx.x = numéro du thread qui va de 0 à la dimension du bloc -1
    //blockId = va correspondre au numéro du bloc qu'on considère 
    //blockDim = dimension du bloc (nombre de cases dans une direction donc si on considère qu'une case = 1 thread, donne le nbr de threads par bloc)
    //gridDim = donne le nombre de blocs dans une grille (n/NbTheadsParBloc)
    //gridDim*blockDim = nombre de threads dans une grille = n

    //va donner par exemple : 0+0*5 (0),..,4+0*5 (4, dernier element du premier bloc),0+1*5 (5),...,4+1*5 (9, dernier element du second bloc)
    curand_init(seed, id_thread, 0, &state[id_thread]);
}

/*
  Fonction qui va servir à effectuer les calculs sur GPU. 
*/

__global__ void kernel(curandState* local_state, float* absorbed, float h, float n, float c, float c_c, float c_s, int* r, int* t, int* b) //uniquement les elements qui sont intialisés dans le main + r b et t
{

  float d = 0; 
  float x = 0; 
  float L;
  float u;
  int i = blockDim.x*blockId.x + threadIdx.x; //sert de compteur 
  //int r_updated, t_updated, b_updated = 0;     

  while (1) {

    u = curand_uniform(&local_state); 
    L = -(1 / c) * log(u);
    x = x + L * cos(d);
    if (x < 0) { //reflechi
      atomicAdd(r,1);//r_updated++;
      break;
    } 
    else if (x >= h) { //transmis
      atomicAdd(t,1);//t_updated++;
      break;
    } 
    else if ((u = curand_uniform(&local_state)) < c_c / c) { //absorbé
      atomicAdd(b,1);//b_updated++;
      absorbed[atomicAdd(j,1)] = x;
      break;
    } 
    else {
      u = curand_uniform(&local_state);
      d = u * M_PI; //direction
    }
  }
  
  //atomicAdd(r,r_updated);
  //atomicAdd(b,b_updated);
  //atomicAdd(t,t_updated); 
}

/*
  Liens utiles dont je me suis servie : 
  - https://stackoverflow.com/questions/16619274/cuda-griddim-and-blockdim
  - https://tcuvelier.developpez.com/tutoriels/gpgpu/cuda/introduction/?page=conclusions
*/

int main(int argc, char *argv[]) {

  // chronometrage
  double start, finish;

  if( argc == 1)
    fprintf( stderr, "%s\n", info);

  float c_c, c_s;
  float h;
  int r, b, t;
  int n, k; 

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
  
  r = b = t = 0;
  
  // affichage des parametres pour verificatrion
  printf("Épaisseur de la plaque : %4.g\n", h);
  printf("Nombre d'échantillons  : %d\n", n);
  printf("C_c : %g\n", c_c);
  printf("C_s : %g\n", c_s);

  float* absorbed_CPU;
  float* absorbed_GPU;
  curandState* devStates;

  /* Définition du nombre de threads et de la taille de la grille */
  dim3 NbThreadsParBloc(128,1,1); dim3 NbBlocks; 
  NbBlocks.x = iDivUp(n,NbThreadsParBloc.x);
  NbBlocks.y = 1;
  NbBlocks.z = 1;

  /* Allocation de la mémoire */
  absorbed_CPU = (float *) calloc(n,sizeof(float)); //sur CPU 
  cudaMalloc((void**) &absorbed_GPU, n*sizeof(float)); //sur GPU
  cudaMalloc (&devStates, n*sizeof(curandState));

  // debut du chronometrage
  start = my_gettimeofday();

  setup_kernel <<<NbBlocks,NbThreadsParBloc.x>>> (devStates,unsigned(time(NULL)));  //initialisation de l'état curandState pour chaque thread 

  for(k=0; k<NbBlocks ; k++) 
  {
    kernel<<<NbBlocks, NbThreadsParBloc.x>>>(&devStates, absorbed_GPU, h, n, c, c_c, c_s, r, t, b); //génération des positions absorbed pour GPU
    //on renvoie aussi r, t et b pour l'affichage plus loin dans le code 
  }

  cudaMemcpy(absorbed_CPU, absorbed_GPU, n*sizeof(float), cudaMemcpyDeviceToHost); //copie du absorbed GPU dans CPU 

  // fin du chronometrage
  finish = my_gettimeofday();

  printf("\nPourcentage des neutrons refléchis : %4.2g\n", (float) r / (float) n);
  printf("Pourcentage des neutrons absorbés : %4.2g\n", (float) b / (float) n);
  printf("Pourcentage des neutrons transmis : %4.2g\n", (float) t / (float) n);

  printf("\nTemps total de calcul: %.8g sec\n", finish - start);
  printf("Millions de neutrons /s: %.2g\n", (double) n / ((finish - start)*1e6));

  cudaFree(absorbed_GPU); 
  cudaFree(devStates);
  free(absorbed_CPU); 

  return EXIT_SUCCESS;
}
