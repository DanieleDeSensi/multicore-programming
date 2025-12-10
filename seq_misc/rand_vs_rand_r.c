// Check the performance of rand() and rand_r() functions
#include "../lec12/my_timer.h"
#include <stdio.h>
#include <stdlib.h>

#define ITER 100000000

int main(int argc, char** argv){
    double start, stop;
    GET_TIME(start);
    double dummy = 0;
    for(int i = 0; i < ITER; i++){
        dummy += rand();
    }
    GET_TIME(stop);
    printf("rand() time: %lf sec\n", stop - start);
    unsigned int s = 0;
    GET_TIME(start);
    for(int i = 0; i < ITER; i++){
        dummy += rand_r(&s);
    }
    GET_TIME(stop);
    printf("rand_r() time: %lf sec\n", stop - start);    
    return 0;
}