// Dylan Donahue
// HPC Homework 2 Problem 1b - Monte Carlo, 02/07/2025

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <time.h>

#define RADIUS 1

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

int main(int argc, char *argv[]){

    if (argc != 3) {
        printf("Usage: %s num_threads, num_darts\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[1]);
    
    long int num_darts = atol(argv[2]);

    omp_set_num_threads(num_threads);
    omp_set_dynamic(0);
    
    long num_darts_in_circle = 0;
    double x, y, distance_squared;

    double start = CLOCK(); 

    #pragma omp parallel 
    {
        unsigned int seed = (unsigned int) clock() + omp_get_thread_num();
    
        #pragma omp for reduction(+:num_darts_in_circle) private(x, y, distance_squared)
        for (long i = 0; i < num_darts; i++) {
            x = (double) rand_r(&seed) / RAND_MAX;
            y = (double) rand_r(&seed) / RAND_MAX;
            distance_squared = x * x + y * y;
            if (distance_squared <= RADIUS) {
                num_darts_in_circle++;
            }
        }
    }

    double pi = 4 * (double) num_darts_in_circle / num_darts;
    double end = CLOCK();

    printf("Estimated value of pi: %f\n", pi);
    double time_spent = end - start;
    printf("Time spent (ms): %f\n", time_spent);

}
