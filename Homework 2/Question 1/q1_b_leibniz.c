// Dylan Donahue
// HPC Homework 2 Problem 1b - Leibniz 02/07/2025

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <time.h>

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

int main(int argc, char *argv[]){

    if (argc != 3) {
        printf("Usage: %s num_threads, num_terms\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[1]);
    long num_terms = atol(argv[2]);

    omp_set_num_threads(num_threads);
    omp_set_dynamic(0);

    double sum = 0;
    double curr_term;

    double start = CLOCK(); 

    #pragma omp parallel for reduction(+ : sum) private(curr_term)
    for (long i=0; i< num_terms; i++){
        curr_term = 1.0/(2.0*i + 1);
        curr_term = (i % 2 == 0) ? curr_term : curr_term * -1;
        sum += curr_term;

    }

    double pi = sum * 4;

    double end = CLOCK();

    printf("Estimated value of pi: %f\n", pi);
    double time_spent = end - start;
    printf("Time spent (ms): %f\n", time_spent); 
    
}
