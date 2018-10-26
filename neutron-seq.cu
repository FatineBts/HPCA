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
  Fonction qui va servir à générer des nombres différents pour chaque thread. 
*/


__device__ float generate(curandState* globalState, int ind) 
{
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform( &localState );
    globalState[ind] = localState;
    return RANDOM;
}

__global__ void kernel(curandState* globalState, float* absorbed, float h, float n, float c, float c_c, float c_s, int* result) //uniquement les elements qui sont intialisés dans le main + r b et t
{
  float L;
  float u;
  int i = blockDim.x*blockIdx.x + threadIdx.x; //sert de compteur 
  int prev; 
  int r_updated = 0, t_updated = 0, b_updated = 0;     
  //r, b, t, j res[0], res[1], res[2], res[3]
  
while(i<n){  
  while (1) {
    float d = 0.0; //direction
    float x = 0.0; //position du neutron
    u = generate(globalState,i); 
    L = -(1 / c) * log(u);
    x = x + L * cos(d);
    if (x < 0) { //reflechi
      r_updated++;
      break;
    } 
    else if (x >= h) { //transmis
      t_updated++;
      break;
    } 
    else if ((u = generate(globalState,i)) < c_c / c) { //absorbé
      b_updated++;
      prev = atomicAdd(result+3,1); //communication interphread pas possible donc on veut l'atomicAdd pour pas écrire de manière concurente (on donne la main à 1 thread) 
      absorbed[prev] = x; //ceci s'applique car on a besoin d'un stockage contigu. On utilise la variable prev car on l'incrementation se fait en 2 étapes donc on doit lui donner le temps
      break;
    } 
    else {
      u = generate(globalState,i);
      d = u * M_PI; //direction
    }
  }
  i += gridDim.x*blockDim.x; //on ajoute le nombre de threads par bloc 

//atomicAdd fait du séquentiel, l'idée est qu'un thread traite plusieurs neutrons et puis quand il a fini, il update r, b et t donc les tableaux. Utiliser cette méthode permet de réduire le nombre d'atomicAdd et donc le temps de calcul
  atomicAdd(result,r_updated); // le pb est qu'on a de l'interaction grace a atomicAdd or CUDA essaye d'éviter cela 
  atomicAdd(result+1,b_updated);
 atomicAdd(result+2,t_updated);
 }//second while
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

  float c, c_c, c_s;
  float h;
  int r, b, t;
  int n,j; 

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
  
  // affichage des parametres pour verificatrion
  printf("Épaisseur de la plaque : %4.g\n", h);
  printf("Nombre d'échantillons  : %d\n", n);
  printf("C_c : %g\n", c_c);
  printf("C_s : %g\n", c_s);

  float* absorbed_CPU;
  float* absorbed_GPU;
  int* result_CPU; 
  int* result_GPU;
  
  curandState* devStates;

  /* Définition du nombre de threads et de la taille de la grille */
  dim3 NbThreadsParBloc(256,1,1); dim3 NbBlocks; 
  NbBlocks.x = iDivUp(n,NbThreadsParBloc.x);
  NbBlocks.y = 1;
  NbBlocks.z = 1;

  /* Allocation de la mémoire */
  absorbed_CPU = (float *) calloc(n,sizeof(float)); //sur CPU 
  cudaMalloc((void**) &absorbed_GPU, n*sizeof(float)); //sur GPU
  cudaMalloc (&devStates, n*sizeof(curandState));
  cudaMalloc((void**) &result_GPU, 4*sizeof(int));
  result_CPU = (int *) calloc(4,sizeof(int)); //sur CPU

  cudaMemcpy(result_CPU,result_GPU,4*sizeof(int), cudaMemcpyHostToDevice); //pour copier result_CPU dans result_GPU
  cudaMemcpy(absorbed_CPU,absorbed_GPU,n*sizeof(float), cudaMemcpyHostToDevice); //pour copier result_CPU dans result_GPU
    
  // debut du chronometrage
  start = my_gettimeofday();

  setup_kernel <<<NbBlocks,NbThreadsParBloc>>> (devStates,unsigned(time(NULL)));  //initialisation de l'état curandState pour chaque thread
  kernel<<<NbBlocks, NbThreadsParBloc>>>(devStates, absorbed_GPU, h, n, c, c_c, c_s, result_GPU); //génération des positions absorbed pour GPU
   //on renvoie aussi r, t et b pour l'affichage plus loin dans le code 
  
  cudaMemcpy(absorbed_CPU, absorbed_GPU, n*sizeof(float), cudaMemcpyDeviceToHost); //copie du absorbed GPU dans CPU 
  cudaMemcpy(result_CPU, result_GPU, 4*sizeof(int), cudaMemcpyDeviceToHost);
  // fin du chronometrage
  finish = my_gettimeofday();

  r = result_CPU[0]; 
  b = result_CPU[1]; 
  t = result_CPU[2]; 
  j = result_CPU[3];

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
     fprintf(f_handle, "%f\n", absorbed_CPU[j]);

  // fermeture du fichier
  fclose(f_handle);
  printf("Result written in " OUTPUT_FILE "\n"); 


  cudaFree(absorbed_GPU); 
  cudaFree(devStates);
  cudaFree(result_GPU);
  free(result_CPU); 
  free(absorbed_CPU);


  return EXIT_SUCCESS;
}
