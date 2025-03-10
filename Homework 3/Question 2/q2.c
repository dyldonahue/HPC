#include <stdio.h>
#include <x86intrin.h>
#include <stdlib.h>
#include <time.h>

 #define N 512

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

void matrix_vector_product(const float matrix[][N], const float *vec, float *result) {
    for (int i = 0; i < N; i++){
        result[i] = 0;
        for (int j = 0; j < N; j++){
            result[i] += matrix[i][j] * vec[j];
        }
    }
}

float matrix_vector_avx512f(const float matrix[][N], const float *vec, float *result) {
   
    for (int i = 0; i < N; i++){

        __m512 sum_vec = _mm512_setzero_ps();

        // iterate through as many groups of 16 as possible, will handle leftovers after
        int j;
        for (j = 0; j + 16 <= N; j += 16){
            __m512 a_vec = _mm512_loadu_ps(&matrix[i][j]); // accessing this way means it must be conitgious/statically allocated
            __m512 b_vec = _mm512_loadu_ps(&vec[j]);
            sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
        }

        // sum floats from AVX reg
        float sum_arr[16];
        _mm512_storeu_ps(sum_arr, sum_vec);

        float total_sum = 0.0f;
        for (int k = 0; k < 16; k++) {
            total_sum += sum_arr[k];
        }

        // handle remainder
        for (int l = j; l < N; l++){
            total_sum += matrix[i][l] * vec[l];
        }

        result[i] = total_sum;
    }
}
   

int main() {

   
    double start, finish, total;
    float a[N][N], b[N], result[N];


    for (int i=0; i< N; i++) {
        b[i] = 1.0;
        result[i] = 0.0;
        for (int j = 0; j < N; j++){
            a[i][j] = 1.0;
        }
    }



    start = CLOCK();
    //matrix_vector_avx512f(a, b, result);
    finish = CLOCK();
    total = finish-start;
    
    printf("Result from a chosen index (Result[76]): %f \n", result[76]);

    printf("The total time for matrix multiplication with AVX = %f ms\n", total);

    start = CLOCK();
    matrix_vector_product(a, b, result);
    finish = CLOCK();
    total = finish-start;
    
    printf("Result from a chosen index (Result[76]): %f \n", result[76]);
    printf("The total time for matrix multiplication without AVX = %f ms\n", total);
    return 0;
}
