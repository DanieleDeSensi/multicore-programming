#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "my_timer.h"

double x[MAX];
double y[MAX];   

int main(int argc, char** argv) {
    int i,iter;
    srand(time(NULL));
    /* Initialize A and x with random values, and y to 0s*/
    for (i = 0; i < MAX; i++) {
        x[i] = (double) rand() / RAND_MAX; // Random number between 0 and 1
        y[i] = (double) rand() / RAND_MAX; // Random number between 0 and 1
    }

    double total_time = 0.0;
    for(iter = 0; iter < ITER; iter++){
        double start, stop;
        GET_TIME(start);
        // Actual code to measure
        for (i = 0; i < MAX; i++){
            x[i] *= 3.0;     
            y[i] *= 3.0;
        }
        GET_TIME(stop);
        total_time += stop-start;
    }

    printf("Average runtime %f usec\n", total_time/ITER);
}
