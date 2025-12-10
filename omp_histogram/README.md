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