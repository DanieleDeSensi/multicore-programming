# Matrix multiplication
This exercise is adapted from the SC08 OpenMP tutorial by Mattson and Meadows (https://www.openmp.org/wp-content/uploads/omp-hands-on-SC08.pdf)
Starting from the serial version of the code (in the *matmul.c* file), parallelize it using OpenMP. The serial code computes the product of two matrices.
We provide several different parallel solutions:
- *matmul_solution.c*: uses a **parallel for** directive to parallelize the outer loop of the matrix multiplication.