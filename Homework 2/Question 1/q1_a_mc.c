// Dylan Donahue
// HPC Homework 2 Problem 1a- Monte Carlo, 02/07/2025

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>

#define RADIUS 1
int num_threads = 0;
int n_per_thread = 0;
long int num_darts = 0;
int* local_thread_sums;


double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

void *throw_dart(void *arg){

    int thread_num = *(int*)arg;
    free(arg);

    int left = thread_num * n_per_thread;
    int right = (thread_num + 1) * n_per_thread - 1;
    unsigned int seed = (unsigned int) clock() + thread_num;
  

     // add remainder to last thread
    if (thread_num == num_threads - 1) {
        right += num_darts % num_threads;
    }

    long num_darts_in_circle = 0;
    double x, y, distance_squared;

    for (long i = left; i < right; i++) {
        x = (double) rand_r(&seed) / RAND_MAX;
        y = (double) rand_r(&seed) / RAND_MAX;
        distance_squared = x * x + y * y;
        if (distance_squared <= RADIUS) {
            num_darts_in_circle++;
        }
    }

    local_thread_sums[thread_num] = num_darts_in_circle;

    return NULL;
}

int main(int argc, char *argv[]){

    if (argc != 3) {
        printf("Usage: %s num_threads, num_darts\n", argv[0]);
        return 1;
    }

    num_threads = atoi(argv[1]);
    num_darts = atol(argv[2]);
    n_per_thread = num_darts / num_threads;

    pthread_t threads[num_threads];

    double start = CLOCK(); 

    local_thread_sums = malloc(sizeof(int) * num_threads);

     for (int i = 0; i < num_threads; i++) {
        int* arg = malloc(sizeof(int));
        *arg = i;
        int ret = pthread_create(&threads[i], NULL, throw_dart, arg);
        if (ret) {
            printf("Error creating thread %d\n", i);
            exit(-1);
        }
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }


  
    int global_sum = 0;
    for (int i =0; i<num_threads;i++){
        global_sum+= local_thread_sums[i];
    }

    free(local_thread_sums);

    double pi = 4 * (double) global_sum / num_darts;
    double end = CLOCK();
    printf("Estimated value of pi: %f\n", pi);

    double time_spent = end - start;
    printf("Time spent (ms): %f\n", time_spent);

    

}