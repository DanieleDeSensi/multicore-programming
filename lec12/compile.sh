#!/bin/bash
# You should use Makefiles instead ;)

# Examples on row-major access
MATRIX_SIZE=10000
NUM_ITERATIONS=10
gcc -Wall -g -O3 -D MAX=${MATRIX_SIZE} -D ITER=${NUM_ITERATIONS} -o cache_01_slow cache_01_slow.c
gcc -Wall -g -O3 -D MAX=${MATRIX_SIZE} -D ITER=${NUM_ITERATIONS} -o cache_01_fast cache_01_fast.c
gcc -Wall -g -O3 -D MAX=${MATRIX_SIZE} -D ITER=${NUM_ITERATIONS} -o cache_01_fast_broken cache_01_fast_broken.c
gcc -Wall -g -O3 -D MAX=${MATRIX_SIZE} -D ITER=${NUM_ITERATIONS} -o cache_01_fast_broken_fixed cache_01_fast_broken_fixed.c

# Examples on loop fusion
MATRIX_SIZE=1000000
NUM_ITERATIONS=1000
gcc -Wall -g -O3 -D MAX=${MATRIX_SIZE} -D ITER=${NUM_ITERATIONS} -o cache_02_slow cache_02_slow.c
gcc -Wall -g -O3 -D MAX=${MATRIX_SIZE} -D ITER=${NUM_ITERATIONS} -o cache_02_fast cache_02_fast.c
