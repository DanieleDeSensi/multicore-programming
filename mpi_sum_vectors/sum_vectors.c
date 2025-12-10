#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int* create_random_vector(int n){
    int* vec = (int*) malloc(n * sizeof(int));
    for(int i = 0; i < n; i++){
        vec[i] = rand() % 10;
    }
    return vec;
}

void print_vector(int* vec, int n){
    for(int i = 0; i < n; i++){
        printf("%d ", vec[i]);
    }
    printf("\n");
}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int *a, *b;
    int n = atoi(argv[1]);
    if(n % size != 0){
        printf("n must be divisible by the number of processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if(rank == 0){
        a = create_random_vector(n);
        b = create_random_vector(n);
        printf("Rank 0: a = ");
        print_vector(a, n);
        printf("Rank 0: b = ");
        print_vector(b, n);
        MPI_Scatter(a, n/size, MPI_INT, MPI_IN_PLACE, n/size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(b, n/size, MPI_INT, MPI_IN_PLACE, n/size, MPI_INT, 0, MPI_COMM_WORLD);
    }else{
        a = (int*) malloc(n/size * sizeof(int));
        b = (int*) malloc(n/size * sizeof(int));
        MPI_Scatter(NULL, n/size, MPI_INT, a, n/size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(NULL, n/size, MPI_INT, b, n/size, MPI_INT, 0, MPI_COMM_WORLD);
    }
    int* c = (int*) malloc(n/size * sizeof(int));
    for(int i = 0; i < n/size; i++){
        c[i] = a[i] + b[i];
    }
    int* c_finale = NULL;
    if(rank == 0){
        c_finale = (int*) malloc(n * sizeof(int));
    }
    MPI_Gather(c, n/size, MPI_INT, c_finale, n/size, MPI_INT, 0, MPI_COMM_WORLD);
    if(rank == 0){
        printf("Rank 0: c = ");
        print_vector(c_finale, n);
    }
    MPI_Finalize();
    return 0;
}