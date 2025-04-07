// Dylan Donahue
// HPC Homework 5 - Question 1 - 03.29.2025

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>


#define NUM_BINS 32
#define MAX_VAL  100000
const int bin_size = MAX_VAL / NUM_BINS; // if not evenly dvisible last bin will slightly higher prob, not a big deal at scale
int N = 0;

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

int main(int argc, char **argv){

    if (argc != 2) {
        printf("Usage: %s num_ints\n", argv[0]);
        return 1;
    }

    N = atoi(argv[1]);

    int bins[NUM_BINS] = {0};

    int num = 0;
    int index = 0;
    int seed = 0;
   double start_time = CLOCK();

    #pragma omp parallel
    seed = time(NULL) + omp_get_thread_num();

    #pragma omp parallel for reduction(+:bins[:NUM_BINS]) private(num, index)
    for (int i = 0; i < N; i++){

        num = (rand_r(&seed) % 100000) + 1;
        index = num / bin_size;
        if (index >= NUM_BINS) index = NUM_BINS - 1;

        bins[index]++; 
    }

    double end_time = CLOCK();
    printf("BIN : # ITEMS IN BIN\n\n");
    for (int i = 0; i < NUM_BINS; i++){
        printf("%d : %d\n", i, bins[i]);
    }

    printf("-------------------------------------\n");
    printf("time spent: %f", end_time - start_time);

    return 0;
}
