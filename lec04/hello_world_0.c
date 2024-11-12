#include <stdio.h>
#include <mpi.h>

int main(void){
    int r = MPI_Init(NULL, NULL);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(r != MPI_SUCCESS){
        printf("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, r);
    }
    char str[256];
    if(rank == 0){
        printf("Hello, World! I am process %d of %d.\n", rank, size);
        int i;
        for(i = 1; i < size; i++){
            
            MPI_Recv(str, 256, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%s", str);
        }
    }else{
        sprintf(str, "Hello, World! I am process %d of %d.\n", rank, size);
        MPI_Send(str, 256, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return 0;
}