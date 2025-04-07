// Dylan Donahue
// HPC Homework 5 - Question 2 - 04.04.2025

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

// Original Stenciling/Nearest Neighbor problem
// ------------------------------------------------------------------------
//float a[n][n][n], b[n][n][n];
//for (i=1; i<n-1; i++)
//    for (j=1; j<n-1; j++)
//        for (k=1; k<n-1; k++) {
//            a[i][j][k]=0.75*(b[i-1][j][k]+b[i+1][j][k]+b[i][j-1][k]
//            + b[i][j+1][k]+b[i][j][k-1]+b[i][j][k+1]);
//        }
// ------------------------------------------------------------------------

#define TILE_SIZE 4

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

__global__ void tiled_stencil(float *a, float *b, int n)
{

    // every thread will need to access indices +/- one, so add a halo edge to entire block, 
    // or else every block will use global mem for all edges. Ex, all threads except T6 and T7 in
    // 2D example below will access global memory for its neighbor, unless we load an extra index all around

    // -------------------------
    // T1   T2    T3     T4 
    // T5   T6    T7     T8
    // T9    T10  T11    T 12
    // ---------------------------
    __shared__ float tiles[TILE_SIZE + 1 + 1][TILE_SIZE + 1 + 1][TILE_SIZE + 1 + 1];

    // 3 dimensional grid, to find unit, take
    // block num * size of tile, then iterate over the threads in that block
    // repeat for all dimentions

    int i = threadIdx.x + blockIdx.x * TILE_SIZE;
    int j = threadIdx.y + blockIdx.y * TILE_SIZE;
    int k = threadIdx.z + blockIdx.z * TILE_SIZE;

    // load the data into (index + 1), so that (index) remains free as a neigbor 
    int ti = threadIdx.x + 1;
    int tj = threadIdx.y + 1;
    int tk = threadIdx.z + 1;


    // flatten operation into one op - ensure less than n -1 llike original stenciling does 

      // find ID of all indices based on where we are within sample space (n)
        // i ~ outer loop === N^2
        // j ~ middle loop === N
        // k ~ inner loop === single iteration

    int id = (i * n * n) + (j * n) + (k);

    // manually load center
    tiles[ti][tj][tk] = b[id];

    // manually compute boundaries (0 and n-1)
    if (threadIdx.x == 0 && i > 0){
        tiles[ti - 1][tj][tk] = b[((i-1) * n * n) + (j * n) + (k)];
    }
    if (threadIdx.y == 0 && j > 0){
        tiles[ti][tj-1][tk] = b[(i * n * n) + ((j-1) * n) + (k)];
    }
    if (threadIdx.z == 0 && k > 0){
        tiles[ti][tj][tk-1] = b[(i * n * n) + (j * n) + (k-1)];
    }

    if (threadIdx.x == TILE_SIZE-1 && i < n-1){
        tiles[ti+1][tj][tk] = b[((i+1) * n * n) + (j * n) + (k)];
    }
    if (threadIdx.y == TILE_SIZE-1 && j < n-1){
        tiles[ti][tj+1][tk] = b[(i * n * n) + ((j+1) * n) + (k)];
    }
    if (threadIdx.z == TILE_SIZE-1 && k < n-1){
        tiles[ti][tj][tk+1] = b[(i * n * n) + (j * n) + (k+1)];
    }

        // perform op - if not on boundary of (0, n-1)

        if (i > 0 && j > 0 && k > 0 && i < n-1 && j < n-1 && k < n-1)
        {

            a[id] = 0.75 * (\
            tiles[ti-1][tj][tk] + 
            tiles[ti+1][tj][tk] + 
            tiles[ti][tj-1][tk] + 
            tiles[ti][tj+1][tk] + 
            tiles[ti][tj][tk-1] + 
            tiles[ti][tj][tk+1]);
        }
}

__global__ void untiled_stencil(float *a, float *b, int n)
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i < n-1 && j < n-1 && k < n-1 && i > 0 && j > 0 && k > 0)
    {

        int id = (i * n * n) + (j * n) + (k);
        int bid_1 = ((i-1) * n * n) + (j * n) + (k);
        int bid_2 = ((i+1) * n * n) + (j * n) + (k);
        int bid_3 = (i * n * n) + ((j-1) * n) + (k);
        int bid_4 = (i * n * n) + ((j+1) * n) + (k);
        int bid_5 = (i * n * n) + (j * n) + (k-1);
        int bid_6 = (i * n * n) + (j * n) + (k+1);
    

        a[id] = 0.75 * \
                (b[bid_1] + 
                 b[bid_2] + 
                 b[bid_3] + 
                 b[bid_4] + 
                 b[bid_5] + 
                 b[bid_6]);
    }
}

int main(int argc, char **argv){

    if (argc != 3) {
        printf("Usage: %s <tiled (1=Yes, 0=No)> <N>\n", argv[0]);
        return 1;
    }
    
    int use_tiled = atoi(argv[1]);
    int n = atoi(argv[2]);

    int bytes = sizeof(float) * n*n*n;
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);


    float *d_a;
    float *d_b;

    double start_time = CLOCK();

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);

    // init b

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);


    dim3 dimBlock(TILE_SIZE, TILE_SIZE, TILE_SIZE);
    dim3 dimGrid(n / TILE_SIZE, n / TILE_SIZE, n / TILE_SIZE);

    if (use_tiled){
        tiled_stencil<<<dimGrid, dimBlock>>>(d_a, d_b, n);
    }

    else untiled_stencil<<<dimGrid, dimBlock>>>(d_a, d_b, n);
 
    // Copy array back to host
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);

    double end_time = CLOCK();

    printf("------------------------------------------\n");
    if (use_tiled) printf("Tiled\n");
    else printf("Untiled\n");
    printf("N = %d\n", n);
    printf("time spent (ms): %f", end_time - start_time);

    cudaFree(d_a);
    cudaFree(d_b);

    free(h_a);
    free(h_b);

    




}