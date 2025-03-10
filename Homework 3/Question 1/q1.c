// Dylan Donahue
// HPC Homework 3 - Question 1 - 02.24.2025

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include <math.h>

typedef enum {
    DOUBLE = 0,
    FLOAT = 1
} data_t;



double double_factorial(double num){
    double *factorials = (double*)malloc(sizeof(double)* (int)(num+1));

    factorials[0] = 1;

    for (double i=1; i <= num; i++){
        factorials[(int)i] = i * factorials[(int)i-1];
    }

    double result = factorials[(int)num];
    free(factorials);
    return result;
}

float float_factorial(float num){
    float *factorials = (float*)malloc(sizeof(float)* (int)(num+1));

    factorials[0] = 1;

    for (float i=1; i <= num; i++){
        factorials[(int)i] = i * factorials[(int)i-1];
    }

    float result = factorials[(int)num];
    free(factorials);
    return result;
}


double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

int main(int argc, char *argv[]){

    if (argc != 3) {
        printf("Usage: %s num_terms {0}double/{1}float\n", argv[0]);
        return 1;
    }

    int num_terms = atoi(argv[1]);

    // if (num_terms > MAX_TERMS){
    //     printf("Too many terms! max is %d", MAX_TERMS);
    //     exit(1);
    // }

    data_t version = atoi(argv[2]);

    // compute taylor series

    if (version == DOUBLE) {

        printf("Computed using doubles...\n\n");

        double sum = 0;
        double vals[4] = {0, 1.45, 3.3, 6.29};

        for (int i = 0; i < 4; i++){

            double start = CLOCK();
            for (int j =0; j < num_terms; j++){
                double exp = 2 * j + 1;
                sum += ( (pow(-1, j) / double_factorial(exp)) * pow(vals[i], exp));

            }

            double end = CLOCK();
            double time_spent = end - start;
            printf("Value: %f\nSin(value): %f\nTime to compute: %f\n", vals[i], sum, time_spent);
            printf("---------------\n");
            if (isnan(sum)) {
                printf("too many terms used, overflowed!\n  ");
                exit(1);
            }
            sum = 0;
        }
    }

    else if (version == FLOAT) {

        printf("Computed using floats...\n\n");

        float sum = 0;
        float vals[4] = {0, 1.45, 3.3, 6.29};

        for (int i = 0; i < 4; i++){

            double start = CLOCK();
            for (int j =0; j < num_terms; j++){
                float exp = 2 * j + 1;
                sum += ( (pow(-1, j) / float_factorial(exp)) * pow(vals[i], exp));

            }

            double end = CLOCK();
            double time_spent = end - start;
            printf("Value: %f\nSin(value): %f\nTime to compute: %f\n", vals[i], sum, time_spent);
            printf("---------------\n");
            if (isnan(sum)) {
                printf("too many terms used, overflowed!\n");
                exit(1);
            }
            sum = 0;
        }

    }

    return 0;

}