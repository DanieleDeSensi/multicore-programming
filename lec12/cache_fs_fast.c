#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "my_timer.h"


#define CLS 16

float data[NUM_THREADS*CLS];

void* thread_fun(void* arg){
    int thread_id = *((int*) arg);
    // Pin 
    cpu_set_t cpuset;
    pthread_t thread = pthread_self();
    CPU_ZERO(&cpuset);
    CPU_SET(thread_id, &cpuset);
    pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset);

    for(int i = 0; i < 100000; i++){
        data[thread_id*CLS] += i;
    }
    return NULL;
}

// Computes matrix-vector multiplication sequentially
int main(int argc, char** argv) {
    int iter;
    srand(time(NULL));
    for(int i = 0; i < NUM_THREADS; i++){
        data[i] = rand();
    }

    int ids[NUM_THREADS];
    for(int i = 0; i < NUM_THREADS; i++){
        ids[i] = i;
    }
    pthread_t threads[NUM_THREADS];

    double total_time = 0.0;
    for(iter = 0; iter < ITER; iter++){
        double start, stop;
        GET_TIME(start);
        // Create threads
        for(int i = 0; i < NUM_THREADS; i++){
            pthread_create(&threads[i], NULL, thread_fun, (void*) &ids[i]);
        }
        
        // Join threads
        for(int i = 0; i < NUM_THREADS; i++){
            pthread_join(threads[i], NULL);
        }
        GET_TIME(stop);
        total_time += stop-start;
    }

    /**
    for (i = 0; i < MAX; i++)
        printf("%f\n", y[i]);
    **/

    printf("Average runtime %f usec\n", total_time/ITER);
}
