Exercises shown during lecture 6.
- rand.c: A simple program in which each program prints a random number. You can see that each rank print the same number, since each one use the same seed. Now, try to add the following code after the MPI_Init:
```c
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(rank);
```
If you run the program again, you will see that each rank print a different number. This is because each rank is using a different seed.

- pi_seq.c: A program that calculates the value of pi (sequentially) using the process we have seen in the slides.
- pi_parallel.c: A program that calculates the value of pi (in parallel) using the process we have seen in the slides.
- argv.c: A program that shows how to use the argv parameter in the main function.