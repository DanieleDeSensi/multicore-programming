#
# K-means 
#
# Parallel computing (Degree in Computer Engineering)
# 2022/2023
#
# (c) 2023 Diego Garcia-Alvarez and Arturo Gonzalez-Escribano
# Grupo Trasgo, Universidad de Valladolid (Spain)
#

# Compilers
CC=gcc
OMPFLAG=-fopenmp
MPICC=mpicc
CUDACC=nvcc

# Flags for optimization and libs
FLAGS=-O3 -Wall
LIBS=-lm

# Targets to build
OBJS=KMEANS_seq KMEANS_omp KMEANS_mpi KMEANS_cuda

# Rules. By default show help
help:
	@echo
	@echo "K-means clustering method"
	@echo
	@echo "Group Trasgo, Universidad de Valladolid (Spain)"
	@echo
	@echo "make KMEANS_seq	Build only the sequential version"
	@echo "make cKMEANS_omp	Build only the OpenMP version"
	@echo "make KMEANS_mpi	Build only the MPI version"
	@echo "make KMEANS_cuda	Build only the CUDA version"
	@echo
	@echo "make all	Build all versions (Sequential, OpenMP)"
	@echo "make debug	Build all version with demo output for small surfaces"
	@echo "make clean	Remove targets"
	@echo

all: $(OBJS)

KMEANS_seq: KMEANS.c
	$(CC) $(FLAGS) $(DEBUG) $< $(LIBS) -o $@

KMEANS_omp: KMEANS_omp.c
	$(CC) $(FLAGS) $(DEBUG) $(OMPFLAG) $< $(LIBS) -o $@

KMEANS_mpi: KMEANS_mpi.c
	$(MPICC) $(FLAGS) $(DEBUG) $< $(LIBS) -o $@

KMEANS_cuda: KMEANS_cuda.cu
	$(CUDACC) $(DEBUG) $< $(LIBS) -o $@


# Remove the target files
clean:
	rm -rf $(OBJS)

# Compile in debug mode
debug:
	make DEBUG="-DDEBUG -g" FLAGS= all

