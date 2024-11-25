// Implements counting sort
// First argument from command line is the maximum value that each element in the array can have.
// e.g., if the argument is 100, then each element of the array can contain a value between 0 and 100.
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define ARRAY_SIZE 100000000

int main(int argc, char** argv){
    int max = atoi(argv[1]);
    int* array = (int*)malloc(ARRAY_SIZE * sizeof(int));
    int* counts = (int*)malloc(max * sizeof(int));
    int* counts_reference = (int*)malloc(max * sizeof(int)); // Compute sequentially and use for error checks
    // Generate random array
    for(unsigned long i = 0; i < ARRAY_SIZE; i++){
        array[i] = rand() % max;        
    }

    // Compute reference counts for error check
    for(unsigned long i = 0; i < max; i++){
        counts_reference[i] = 0;
    }
    for(unsigned long i = 0; i < ARRAY_SIZE; i++){
        counts_reference[array[i]]++;
    }

    double start = omp_get_wtime();
    for(unsigned long i = 0; i < max; i++){
        counts[i] = 0;
    }

    #pragma omp parallel for
    for(unsigned long i = 0; i < ARRAY_SIZE; i++){
        #pragma omp critical
        counts[array[i]]++;
    }
    double stop = omp_get_wtime();

    for(unsigned long i = 0; i < max; i++){
        if(counts[i] != counts_reference[i]){
            fprintf(stderr, "Error: counts[%lu] = %d, counts_reference[%lu] = %d\n", i, counts[i], i, counts_reference[i]);
            return 1;
        }
        printf("%d elements with value %ld\n", counts[i], i);
    }
    printf("Total runtime: %f secs\n", stop - start);

    free(counts_reference);    
    free(counts);
    free(array);
    return 0;
}