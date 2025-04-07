// Dylan Donahue
// HPC Homework 4 - Question 1a - 03.12.2025

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <unistd.h>


int main(int argc, char **argv){

    int rank, size;
    int number  = 1;
    char host[256];

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    gethostname(host, sizeof(host));

    if (rank == 0){
        printf("Process %d on node %s is printing... number is now equal to %d\n", rank + 1, host, number);
        number++;
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    }

    else if (rank <= 63){
        MPI_Recv(&number, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // since we want to print 1-64, going to do rank + 1
        printf("Process %d on node %s is printing... number is now equal to %d\n", rank + 1, host, number);
        number++;
        if (rank < 63){
            MPI_Send(&number, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
        }
        
    }

    MPI_Finalize();
    return 0;
}