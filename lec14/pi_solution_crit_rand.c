
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

static long num_trials = 100000000;

double get_rand_minus_one_one(){
    return 2 * (rand() / (double)RAND_MAX) - 1;
}

int main ()
{
   long i;  long Ncirc = 0;
   double pi, x, y, test, total_time;
   double r = 1.0;   // radius of circle. Side of squrare is 2*r 
   srand(time(NULL));

   total_time = omp_get_wtime();
   #pragma omp parallel
   {

      #pragma omp single
          printf(" %d threads ",omp_get_num_threads());

      #pragma omp for private(x,y,test)
      for(i=0;i<num_trials; i++)
      {
         x = get_rand_minus_one_one(); 
         y = get_rand_minus_one_one();

         test = x*x + y*y;

         if (test <= r*r){
            #pragma omp critical
            Ncirc++;
         }
       }
    }

    pi = 4.0 * ((double)Ncirc/(double)num_trials);

    printf("\n %ld trials, pi is %f ",num_trials, pi);
    printf(" in %f seconds\n",omp_get_wtime()-total_time);

    return 0;
}