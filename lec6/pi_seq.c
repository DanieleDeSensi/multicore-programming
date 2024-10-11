#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double get_rand_minus_one_one(){
    return 2 * (rand() / (double)RAND_MAX) - 1;
}

int main(int argc, char** argv){
    int num_tosses = atoi(argv[1]);
    srand(time(NULL));
    int toss;
    int num_hits = 0;
    for(toss = 0; toss < num_tosses; toss++){
        double x = get_rand_minus_one_one();
        double y = get_rand_minus_one_one();
        if(x*x + y*y <= 1){
            num_hits++;
        }
    }
    double pi_estimate = 4 * num_hits / ((double)num_tosses);
    printf("Estimate of pi = %f\n", pi_estimate);
    return 0;
}