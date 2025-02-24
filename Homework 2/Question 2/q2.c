// Dylan Donahue
// HPC Homework 2 Problem 2 - 01/31/2025

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

typedef struct {
    int id;
    pthread_mutex_t *forks;
    int num_philosophers;
} philosopher_args;


void *philosopher(void *arg) {
    
    philosopher_args *args = (philosopher_args*) arg;
    int id = args->id;
    int num_philosophers = args->num_philosophers;
    pthread_mutex_t *forks = args->forks;

    // allow all threads to catch up
    sleep(1);

    printf("Philosopher %d is thinking\n", id);

    // left fork = id
    // right fork = (id + 1) % num_philosophers (wrap around for last philosopher)

    // avoid deadlock by ensuring not everyone picks up the left fork first
    // if that happened, no one would be able to pick up the right fork

    if (id % 2 == 0) {
        pthread_mutex_lock(&forks[(id + 1) % num_philosophers]);
        printf("Philosopher %d picked up their right fork\n", id);
        pthread_mutex_lock(&forks[id]);
        printf("Philosopher %d picked up their left fork\n", id);
    } else {
        pthread_mutex_lock(&forks[id]);
        printf("Philosopher %d picked up their left fork\n", id);
        pthread_mutex_lock(&forks[(id + 1) % num_philosophers]);
        printf("Philosopher %d picked up their right fork\n", id);
    }

    printf("Philosopher %d is eating\n", id);

    // generate random eat time
    int eat_time = rand() % 100000;
    usleep(eat_time);


    printf("Philosopher %d is done eating\n", id);
    pthread_mutex_unlock(&forks[id]);
    pthread_mutex_unlock(&forks[(id + 1) % num_philosophers]);

    return NULL;

}

int main(int argc, char *argv[]) {
    // Check for correct number of arguments
    if (argc != 2) {
        printf("Usage: %s <number of philosophers>\n", argv[0]);
        return 1;
    }

    // Get number of philosophers
    int num_philosophers = atoi(argv[1]);

    if (num_philosophers % 2 == 0) {
        printf("Number of philosophers must be odd\n");
        return 1;
    }

    pthread_t philosophers[num_philosophers];
    pthread_mutex_t forks[num_philosophers];

    for (int i = 0; i < num_philosophers; i++) {
        pthread_mutex_init(&forks[i], NULL);
    }

    philosopher_args args[num_philosophers];

    for (int i = 0; i < num_philosophers; i++) {
        args[i].id = i;
        args[i].forks = forks;
        args[i].num_philosophers = num_philosophers;
        pthread_create(&philosophers[i], NULL, philosopher, &args[i]);
    }

    for (int i = 0; i < num_philosophers; i++) {
        pthread_join(philosophers[i], NULL);
    }

    return 0;
}
