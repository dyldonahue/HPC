// Dylan Donahue
// HPC Homework 2 Problem 1a - Leibniz 02/07/2025

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>

int num_threads = 0;
int n_per_thread = 0;
long int num_terms = 0;
double* local_thread_sums;

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

void *thread_summation(void *arg){

    int thread_num = *(int*)arg;
    free(arg);

    int left = thread_num * n_per_thread;
    int right = (thread_num + 1) * n_per_thread - 1;

     // add remainder to last thread
    if (thread_num == num_threads - 1) {
        right += num_terms % num_threads;
    }

    double local_sum = 0;
    double curr_term;

    for (long i=left; i< right; i++){
        curr_term = 1.0/(2.0*i + 1);
        curr_term = (i % 2 == 0) ? curr_term : curr_term * -1;
        local_sum += curr_term;
    }

    local_thread_sums[thread_num] = local_sum;

    return NULL;

}

int main(int argc, char *argv[]){

    if (argc != 3) {
        printf("Usage: %s num_threads, num_terms\n", argv[0]);
        return 1;
    }

    num_threads = atoi(argv[1]);
    num_terms = atol(argv[2]);
    n_per_thread = num_terms / num_threads;

    pthread_t threads[num_threads];

    double start = CLOCK(); 

    local_thread_sums = malloc(sizeof(double) * num_threads);

     for (int i = 0; i < num_threads; i++) {
        int* arg = malloc(sizeof(int));
        *arg = i;
        int ret = pthread_create(&threads[i], NULL, thread_summation, arg);
        if (ret) {
            printf("Error creating thread %d\n", i);
            exit(-1);
        }
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    double global_sum = 0;

    for (int i=0; i< num_threads; i++){
        global_sum += local_thread_sums[i];
    } 

    double pi = global_sum * 4;
    double end = CLOCK();

    printf("Estimated value of pi: %f\n", pi);
    double time_spent = end - start;
    printf("Time spent (ms): %f\n", time_spent); 
    
}
