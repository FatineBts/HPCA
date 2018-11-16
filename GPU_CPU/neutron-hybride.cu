#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <curand.h> //bibliothèque pour la generation de nombres aléatoires  
#include <curand_kernel.h>

#define OUTPUT_FILE "/tmp/3302011/absorbed.dat"
#define NbpaquetN 512

char info[] = "\
Usage:\n\
    neutron-gpu H Nb C_c C_s\n\
\n\
    H  : épaisseur de la plaque\n\
    Nb : nombre d'échantillons\n\
    C_c: composante absorbante\n\
    C_s: componente diffusante\n\
\n\
Exemple d'execution : \n\
    neutron-gpu 1.0 500000000 0.5 0.5\n\
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
    float random = curand_uniform(&localState);
    globalState[ind] = localState;
    return random;
}

__global__ void kernel(curandState* globalState, float* absorbed, float h, int n, float c, float c_c, float c_s, int paquetN, int* result) //uniquement les elements qui sont intialisés dans le main + r b et t
{
  float d; 
  float x; 
  float L;
  float u;
  int i = blockDim.x*blockIdx.x + threadIdx.x; //sert de compteur
  int gi = i;
  int prev;
  __shared__ int r_local[NbpaquetN]; //taille du nombre de threads qu'il va nous falloir pour traiter paquetN neutrons par thread, comme on a imposé 512 threads par bloc dans le main, NbpaquetN vaudra 512, on aura donc 512 threads qui vont traiter chacun 1 paquetN donc 512 paquetN. Il s'agit du nombre de paquetN.  
  __shared__ int t_local[NbpaquetN]; 
  __shared__ int b_local[NbpaquetN];
  r_local[threadIdx.x] = 0; //on initialise à zéro le tableau
  t_local[threadIdx.x] = 0; 
  b_local[threadIdx.x] = 0; 
  //int r_updated = 0, t_updated = 0, b_updated = 0;  
  int r, t, b; 

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
  i += (gridDim.x*blockDim.x); //nombre de threads dans une grille qui correspond ici à un bloc, on fait des sauts correspondants aux nombres de threads dans un bloc ce qui donne 512 
}//while(i<n) //tant qu'on a pas traité tous les neutrons 

 r_local[threadIdx.x] = r; //on initialise à zéro le tableau
 t_local[threadIdx.x] = t; 
 b_local[threadIdx.x] = b; 
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
  int paquetN;
  
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
  if (argc > 5)
     paquetN = atof(argv[5]);
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
  //il s'agit du nombre de neutrons traités par 1 thread = nombres de neutrons dans un paquet. 
  curandState* devStates;
  printf("paquetN : %d\n", paquetN);

  /* Définition du nombre de threads et de la taille de la grille */
  dim3 NbThreadsParBloc(NbpaquetN,1,1); dim3 NbBlocks;  

  NbBlocks.x = NbpaquetN;//iDivUp(iDivUp(n,paquetN),NbThreadsParBloc.x); //on fait en sorte qu'au lieu qu'un thread traite 1 neutron, 1 thread va en traiter paquetN. On impose le nombre de threads par blocs à 512 et on cherche le nombre de blocs qu'il faudrait si on a n neutrons avec un paquet de neutrons traité par 1 thread égal à paquetN. Plus on augmente paquetN et plus n est petit et à priori plus la vitesse d'execution devrait être élevée.   
  

  //printf("n/paquetN %d\n",iDivUp(n,paquetN));
  printf("nombre de blocs %4.2d\n",NbBlocks.x);
  NbBlocks.y = 1;
  NbBlocks.z = 1;  

  /* Allocation de la mémoire */
  absorbed_CPU = (float *) calloc(n,sizeof(float)); //sur CPU 
  result_CPU = (int *) calloc(4,sizeof(int)); //sur CPU
  cudaMalloc((void**) &absorbed_GPU, n*sizeof(float)); //sur GPU
  cudaMalloc (&devStates, NbThreadsParBloc.x*NbBlocks.x*sizeof(curandState));
  cudaMalloc((void**) &result_GPU, 4*sizeof(int));

  cudaMemcpy(absorbed_CPU, absorbed_GPU, n*sizeof(float), cudaMemcpyHostToDevice); //copie du absorbed CPU dans GPU 
  cudaMemcpy(result_CPU, result_GPU, 4*sizeof(int), cudaMemcpyHostToDevice);

  // debut du chronometrage
  start = my_gettimeofday();


  #pragma omp parallel 
  {
  #pragma omp master
  {
  setup_kernel <<<NbBlocks,NbThreadsParBloc>>> (devStates,unsigned(time(NULL)));  //initialisation de l'état curandState pour chaque thread
  kernel<<<NbBlocks, NbThreadsParBloc>>>(devStates, absorbed_GPU, h, n, c, c_c, c_s, paquetN, result_GPU); //génération des positions absorbed pour GPU
   //on renvoie aussi r, t et b pour l'affichage plus loin dans le code 
  
  cudaMemcpy(absorbed_CPU, absorbed_GPU, n*sizeof(float), cudaMemcpyDeviceToHost); //copie du absorbed GPU dans CPU 
  cudaMemcpy(result_CPU, result_GPU, 4*sizeof(int), cudaMemcpyDeviceToHost);


 }//fin omp master 

  }//fin du pragma omp parallel


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

  printf("réfléchis = %d, absorbés = %d, transmis = %d\n", r, b,t);
  printf("Total traité: %d\n", r + b +t);

/*
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
*/

  cudaFree(absorbed_GPU); 
  cudaFree(devStates);
  cudaFree(result_GPU);
  free(result_CPU); 
  free(absorbed_CPU);


  return EXIT_SUCCESS;
}
