# Pi calculation
This exercise is adapted from the SC08 OpenMP tutorial by Mattson and Meadows (https://www.openmp.org/wp-content/uploads/omp-hands-on-SC08.pdf)
Starting from the serial version of the code (in the *pi.c* file), parallelize it using OpenMP. The serial code computes pi using the Monte Carlo method we
have also seen in the MPI and Pthreads examples.
We provide several different parallel solutions:
- *pi_solution_crit_rand.c*: uses a **critical section** to update the global variable that stores the number of points inside the circle.
- *pi_solution_crit.c*: uses a **critical section** to update the global variable that stores the number of points inside the circle, but uses **rand_r** instead of **rand**
  to generate random number. This provides a significant speedup, since rand uses a mutex to protect its internal state. You can check that rand_r is usually
  faster than rand by running the rand_vs_rand_r.c example.
- *pi_solution_ato.c*: uses **atomic operations** to update the global variable that stores the number of points inside the circle.
- *pi_solution_red.c*: uses a **reduction** to update the global variable that stores the number of points inside the circle.