#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv){
    MPI_Init(NULL, NULL);
    printf("Rand %d\n", rand());
    MPI_Finalize();
    return 0;
}
