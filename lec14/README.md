Today we are going to see the following exercises using OpenMP.

# Matrix multiplication
This exercise is adapted from the SC08 OpenMP tutorial by Mattson and Meadows (https://www.openmp.org/wp-content/uploads/omp-hands-on-SC08.pdf)
Starting from the serial version of the code (in the *matmul.c* file), parallelize it using OpenMP. The serial code computes the product of two matrices.
We provide several different parallel solutions:
- *matmul_solution.c*: uses a **parallel for** directive to parallelize the outer loop of the matrix multiplication.

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

# Histogram calculation
Starting from the serial version of the code (in the *histogram.c* file), parallelize it using OpenMP. The serial code computes the histogram of the values in an array.
The code takes one command line argument, representing the maximum value that the numbers in the array can have.
We provide several different parallel solutions:
- *histogram_solution_trivial_crit.c*: uses a **critical section** to update the histogram.
- *histogram_solution_trivial_ato.c*: uses **atomic operations** to update the histogram.
- *histogram_solution_ato_local.c*: uses a **reduction** to update the histogram, but avoids (part of) the false sharing, by accumulating into a local array, and then
    updating the global histogram array using atomics.
- *histogram_solution_ato_local_better.c*: uses a **reduction** to update the histogram, and avoid all the false sharing, by accumulating into a local array, and then
    updating the global histogram array using atomics. Those local arrays are made bigger so to avoid false sharing.
- *histogram_solution_red.c*: uses a **reduction** to update the histogram.