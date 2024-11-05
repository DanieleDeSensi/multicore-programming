#!/bin/bash
# You should use Makefiles instead ;)

# Examples on row-major access
MATRIX_SIZE=10000
NUM_ITERATIONS=10
OPT=-O3

gcc -Wall -g ${OPT} -D MAX=${MATRIX_SIZE} -D ITER=${NUM_ITERATIONS} -o cache_01_slow cache_01_slow.c
gcc -Wall -g ${OPT} -D MAX=${MATRIX_SIZE} -D ITER=${NUM_ITERATIONS} -o cache_01_fast cache_01_fast.c
gcc -Wall -g ${OPT} -D MAX=${MATRIX_SIZE} -D ITER=${NUM_ITERATIONS} -o cache_01_fast_broken cache_01_fast_broken.c
gcc -Wall -g ${OPT} -D MAX=${MATRIX_SIZE} -D ITER=${NUM_ITERATIONS} -o cache_01_fast_broken_fixed cache_01_fast_broken_fixed.c

# Examples on false sharing
NUM_ITERATIONS=1000
OPT=-O0
NUM_THREADS=4
gcc -Wall -g ${OPT} -D NUM_THREADS=${NUM_THREADS} -D MAX=${MATRIX_SIZE} -D ITER=${NUM_ITERATIONS} -o cache_fs_slow cache_fs_slow.c -pthread
gcc -Wall -g ${OPT} -D NUM_THREADS=${NUM_THREADS} -D MAX=${MATRIX_SIZE} -D ITER=${NUM_ITERATIONS} -o cache_fs_fast cache_fs_fast.c -pthread

# Examples on branch prediction
MATRIX_SIZE=1000000
NUM_ITERATIONS=10
OPT=-O0
gcc -Wall -g ${OPT} -D MAX=${MATRIX_SIZE} -D ITER=${NUM_ITERATIONS} -o branch_prediction_slow branch_prediction_slow.c
gcc -Wall -g ${OPT} -D MAX=${MATRIX_SIZE} -D ITER=${NUM_ITERATIONS} -o branch_prediction_fast branch_prediction_fast.c
