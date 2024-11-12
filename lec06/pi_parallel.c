#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

double get_rand_minus_one_one(){
    return 2 * (rand() / (double)RAND_MAX) - 1;
}

int main(int argc, char** argv){
    int num_tosses = atoi(argv[1]);
    int toss;
    int num_hits = 0;
    MPI_Init(NULL, NULL);
    double start_time = MPI_Wtime();
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(time(NULL)*rank);
    int local_tosses = num_tosses / world_size;
    for(toss = 0; toss < local_tosses; toss++){
        double x = get_rand_minus_one_one();
        double y = get_rand_minus_one_one();
        if(x*x + y*y <= 1){
            num_hits++;
        }
    }
    int total_hits;
    MPI_Reduce(&num_hits, &total_hits, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0){
        double pi_estimate = 4 * total_hits / ((double)num_tosses);
        printf("Estimate of pi = %f Computed in %f seconds\n", pi_estimate, MPI_Wtime() - start_time);
    }
    MPI_Finalize();
    return 0;
}