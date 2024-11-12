#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("I am rank %d and argv[0] is %s argv[1] is %s\n", 
           rank, argv[0], argv[1]);
    MPI_Finalize();
    return 0;
}