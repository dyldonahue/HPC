#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define N 512
#define LOOPS 10

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

long get_l1_cache(){
    long cache_size;

    // not sure what system this will be compiled on, so going to assume its a linux flavor of some sort and grab cache size

    FILE *file = fopen("/sys/devices/system/cpu/cpu0/cache/index0/size", "r");
    int ret = 0;
    if (file) {
        ret = fscanf(file, "%zu", &cache_size);
        if (ret != 1) ret = -1;
        fclose(file);
    }
    if (ret == -1) return ret;
    else return cache_size;
}

int main()
{
    // will be in kb, proabbly 32k. Will set to 32k in case error
    long cache = get_l1_cache();
    if (cache == -1) cache = 32;

    int block_size = cache / 2; 
    
    double a[N][N]; /* input matrix */
    double b[N][N]; /* input matrix */
    double c[N][N]; /* result matrix */
    int i,j,k,l, num_zeros;
    int kk, jj;
    double start, finish, total;
    double sum;

    

    /* initialize a dense matrix */
    for(i=0; i<N; i++){
        for(j=0; j<N; j++){
            a[i][j] = (double)(i+j);
            b[i][j] = (double)(i-j);
        }
    }

    /* tiled approach */
    printf("starting dense matrix multiply tiled\n");
    start = CLOCK();

    for (l = 0; l < LOOPS; l++) {
        // load balance and spread across 3 levels if possible
          #pragma omp parallel for schedule(dynamic, block_size) private(l, i, j, k, jj, kk, sum) collapse(3)
        for (kk = 0; kk < N; kk += block_size) {  
            for (jj = 0; jj < N; jj += block_size) {  
                for (i = 0; i < N; i++) {  
                    for (j = jj; j < jj + block_size; j++) {  
                        if (jj ==0 && kk==0) c[i][j] = 0;
                        sum = 0.0;

                        // vectorize this into fewer loops if supported
                        #pragma omp simd 
                        for (k = kk; k < kk + block_size; k++) {  
                            sum += a[i][k] * b[k][j];  
                            
                        }
                        c[i][j] += sum;  
                        
                    }
                }
            }
        }
    }


    finish = CLOCK();
    total = finish-start;
    printf("a result %g \n", c[7][8]); /* prevent dead code elimination */
    printf("The total time for matrix multiplication with dense matrices= %f ms\n",
    total);
    
    
    // OLD VERSION - AS GIVEN TO US
      /* initialize a dense matrix */
    // for(i=0; i<N; i++){
    //     for(j=0; j<N; j++){
    //         a[i][j] = (double)(i+j);
    //         b[i][j] = (double)(i-j);
    //     }
    // }


    // printf("starting dense matrix multiply \n");
    // start = CLOCK();

    // for (l=0; l<LOOPS; l++) {
        
    //     for(i=0; i<N; i++)
    //         for(j=0; j<N; j++){
    //             c[i][j] = 0.0;
    //             for(k=0; k<N; k++){
    //                 c[i][j] = c[i][j] + a[i][k] * b[k][j];
    //             }
    //         }
    // }
    // finish = CLOCK();
    // total = finish-start;
    // printf("a result %g \n", c[7][8]); /* prevent dead code elimination */
    // printf("The total time for matrix multiplication with dense matrices = %f ms\n",
    // total);

    /* CSR approach */

    /* initialize a sparse matrix */

    num_zeros = 0;
    for(i=0; i<N; i++){
        for(j=0; j<N; j++){
            if ((i<j)&&(i%2>0))
            {
                a[i][j] = (double)(i+j);
                b[i][j] = (double)(i-j);
            }
            else
            {
                num_zeros++;
                a[i][j] = 0.0;
                b[i][j] = 0.0;
            }
        }
    }


    // create CSR arrays

    // based on code above, i < j 50% of time, and i %2 ==0 50% of time, so 25% will be non zero

    double a_non_zero[(N* N / 4) + 1];
    int a_col_non_zero[(N*N / 4) + 1];
    int a_row_idx[N+1];

    a_row_idx[0] = 0;
    int curr_a_pos =0;


    printf("starting sparse matrix multiply \n");
    start = CLOCK();

    for(i=0; i<N; i++){
        for(j=0; j<N; j++){
            if (a[i][j]!= 0){
                a_non_zero[curr_a_pos] = a[i][j];
                a_col_non_zero[curr_a_pos] = j;
                curr_a_pos++;
            }
        }
        a_row_idx[i+1] = curr_a_pos;
    }


    // init to zeroes

    for(i=0; i<N; i++){
        for(j=0; j<N; j++){
            c[i][j] = 0;
        }
    }

    // multiply the CSR A by B
    #pragma omp parallel for
    for (l=0; l<LOOPS; l++) {
        for (int i = 0; i < N; i++) { // Iterate over rows of A
            for (int k = a_row_idx[i]; k < a_row_idx[i + 1]; k++) {
                int colA = a_col_non_zero[k];  // Column index of non-zero in A
                int valA = a_non_zero[k];   // Non-zero value in A

                // Multiply non-zero A[i, colA] with entire row of B[colA]
                for (int j = 0; j < N; j++) {
                    if (j ==0) c[i][j] = 0;
                    c[i][j] += valA * b[colA][j];
                }
            }
        }
    }
            
    finish = CLOCK();
    total = finish-start;
    printf("A result %g \n", c[7][8]); /* prevent dead code elimination */
    printf("The total time for matrix multiplication with sparse matrices = %f ms\n",
    total);
    printf("The sparsity of the a and b matrices = %f \n", (float)num_zeros/(float)
    (N*N));


    /* ORIGINAL IMPLMENTATION AS PROVIDED TO US */
    /* initialize a sparse matrix */

    // num_zeros = 0;
    // for(i=0; i<N; i++){
    //     for(j=0; j<N; j++){
    //         if ((i<j)&&(i%2>0))
    //         {
    //             a[i][j] = (double)(i+j);
    //             b[i][j] = (double)(i-j);
    //         }
    //         else
    //         {
    //             num_zeros++;
    //             a[i][j] = 0.0;
    //             b[i][j] = 0.0;
    //         }
    //     }
    // }
    // printf("starting sparse matrix multiply \n");
    // start = CLOCK();
    // for (l=0; l<LOOPS; l++) {
    //     for(i=0; i<N; i++)
    //         for(j=0; j<N; j++){
    //             c[i][j] = 0.0;
    //             for(k=0; k<N; k++)
    //                 c[i][j] = c[i][j] + a[i][k] * b[k][j];
    //         }
    // }
    // finish = CLOCK();
    // total = finish-start;
    // printf("A result %g \n", c[7][8]); /* prevent dead code elimination */
    // printf("The total time for matrix multiplication with sparse matrices = %f ms\n",
    // total);
    // printf("The sparsity of the a and b matrices = %f \n", (float)num_zeros/(float)
    // (N*N));
    return 0;
}