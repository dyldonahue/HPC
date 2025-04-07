// Dylan Donahue
// HPC Homework 4 - Question 2- 03.14.2025

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <unistd.h>
#include <math.h>



#define NUM_BINS 32
#define N        8000000
#define MAX_VAL  100000
const int bin_size = MAX_VAL / NUM_BINS; // if not evenly dvisible last bin will slightly higher prob, not a big deal at scale

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

int main(int argc, char **argv){

    int local_count[NUM_BINS] = {0};

    double start_time = CLOCK();

    MPI_Init(NULL, NULL);

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    unsigned int seed = time(NULL) + rank;

    int start = (N/size) * rank;
    int end = (N/size) * (rank+1);

    for (int i = start; i < end; i++){
        int num = (rand_r(&seed) % 100000) + 1;

        int index = num / bin_size;

        if (index >= NUM_BINS) index = NUM_BINS -1;

        local_count[index]++;
    }

    int global_count[NUM_BINS] = {0};
    MPI_Reduce(local_count, global_count, NUM_BINS, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0){

        double end_time = CLOCK();
        printf("BIN : # ITEMS IN BIN\n\n");
        for (int i = 0; i < NUM_BINS; i++){
            printf("%d : %d\n", i, global_count[i]);
        }

        printf("-------------------------------------\n");
        printf("time spent: %f", end_time - start_time);
    }

    MPI_Finalize();

    return 0;
}
