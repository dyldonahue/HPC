// Dylan Donahue
// HPC Homework 5 - Question 1 - 03.29.2025

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

#define NUM_BINS 32
#define MAX_VAL  100000
const int bin_size = MAX_VAL / NUM_BINS; // if not evenly dvisible last bin will slightly higher prob, not a big deal at scale

int N = 0;

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

__global__ void bin(int* data_array, int* class_array, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n){
        int value = data_array[id];

        // stores the class/bin num that data @ index id belongs to 
        class_array[id] = value / bin_size;
    }
  
}
 

int main(int argc, char **argv){

            if (argc != 2) {
                printf("Usage: %s num_ints\n", argv[0]);
                return 1;
            }

            N = atoi(argv[1]);

            // host data + class arrays
            int *h_data = (int*)malloc(sizeof(int) * N);
            int *h_class = (int*)calloc(N, sizeof(int));

    // device data + class arrays
    int *d_data;
    int *d_class;
    cudaMalloc(&d_data, sizeof(int) * N);
    cudaMalloc(&d_class, sizeof(int) * N);

    //iniitalze random data

    srand(time(NULL));

    for (int i=0; i < N; i++){
        h_data[i] = (rand() % 100000) + 1;
    }

    double start_time = CLOCK();

    cudaMemcpy(d_data, h_data, sizeof(int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_class, h_class, sizeof(int)*N, cudaMemcpyHostToDevice);

    int blockSize, gridSize;
 
    // Number of threads in each thread block
    blockSize = 1024;
 
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)N/blockSize);
  
    bin<<<gridSize, blockSize>>>(d_data, d_class, N);
 
    // Copy array back to host
    cudaMemcpy(h_class, d_class, sizeof(int)*N, cudaMemcpyDeviceToHost);

    // reduction

    int bins[NUM_BINS] = {0};
    for (int i = 0; i < N; i++){
        bins[h_class[i]]++;
    }
   
    double end_time = CLOCK();

    // print one value from each class
    for (int i =0; i < NUM_BINS; i++){
        for (int j = 0; j < N; j++){
            if (h_class[j] == i){
                printf("Class %d value: %d\n", i, h_data[j]);
                break;
            }
        }
    }

    printf("-------------------------------------\n");
    printf("N: %d\n", N);
    printf("block size: %d\n", blockSize);
    printf("time spent (ms): %f", end_time - start_time);
 
    cudaFree(d_data);
    cudaFree(d_class);

    free(h_data);
    free(h_class);

    return 0;
}