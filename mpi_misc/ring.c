#include <stdio.h>
#include <mpi.h>

int main(void){
    int rank, size;
    int send_right = 19;
    int send_left = 23;
    int recv_left, recv_right;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Request requests[4];
    // Send right   
    MPI_Isend(&send_right, 1, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD, &requests[0]);
    // Send left
    MPI_Isend(&send_left, 1, MPI_INT, (rank - 1 + size) % size, 0, MPI_COMM_WORLD, &requests[1]);
    // Recv from right
    MPI_Irecv(&recv_right, 1, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD, &requests[2]);
    // Recv from left
    MPI_Irecv(&recv_left, 1, MPI_INT, (rank - 1 + size) % size, 0, MPI_COMM_WORLD, &requests[3]);
    // Compute anything
    // ...
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    MPI_Finalize();
    return 0;
}
