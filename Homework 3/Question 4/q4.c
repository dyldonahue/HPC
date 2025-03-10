#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>

#define N 256

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}


int main() {

  
    float A[N*N], B[N*N], C[N*N];
    
    // init A + B
    for(int i = 0; i < N; i++){
        for (int j= 0; j < N; j++){
            A[i*N + j] = (float)(i+j);
            B[i*N + j] = (float)(i-j);
        }

    }

    double start = CLOCK();
    // Perform matrix multiplication using cblas_sgemm
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);

    double finish = CLOCK();
    double total = finish-start;

    // print a random result (same one as printed in Q3)
    printf("A random result: %f \n", C[7*N + 8]);
    printf("The total time for matrix multiplication with OpenBLAS= %f ms\n", total);

    return 0;
}
