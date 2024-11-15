In this exercise, you will parallelize a sequential program that applies the Black-Scholes option pricing formula to a large dataset. The Black-Scholes formula is a mathematical model for the dynamics of a financial market containing derivative investment instruments. The formula calculates the price of a financial option comprising a stock and an option to buy or sell the stock at a specified price at a future date. 
Regardless of the specific details of the formula, in this exercise you are supposed to parallelize the sequential code available in the file `blackscholes.c`. 
The code reads a dataset from a file and applies the Black-Scholes formula to each record in the dataset. Sample input datasets can be found in the `inputs` directory. The code writes the results to an output file.
You can generate new datasets (e.g., if you want to generate bigger datasets), by running the `inputgen.c` program. The prorgram takes two arguments: the number of records in the dataset and the output file.
You can compile all the code available in this directory by running the `compile.sh` script. 

The `blackscholes.c` application takes two arguments from command line: the input file and the output file. The program runs some correctness checks. You should implement two parallel versions of this application, one using OpenMP and another using Pthreads. The parallel versions should read the input file and write the output file in the same format as the sequential version. The parallel versions should also produce the same results as the sequential version (if not, the program will print error messages).

In the `solution` folder, you will find the proposed solutions.

The code for both the sequential version and the parallel solution has been adapted from the PARSEC benchmark.