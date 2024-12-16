#
# Exact genetic sequence alignment
#
# Parallel computing (Degree in Computer Engineering)
# 2023/2024
#
# (c) 2024 Arturo Gonzalez-Escribano
# Grupo Trasgo, Universidad de Valladolid (Spain)
#

# Compilers
CC=gcc
OMPFLAG=-fopenmp
MPICC=mpicc
CUDACC=nvcc

# Flags for optimization and external libs
LIBS=-lm
FLAGS=-O3 -Wall
CUDAFLAGS=-O3 -Xcompiler -Wall

# Targets to build
OBJS=align_seq align_omp align_mpi align_cuda

# Rules. By default show help
help:
	@echo
	@echo "Exact genetic sequence alignment"
	@echo
	@echo "Group Trasgo, Universidad de Valladolid (Spain)"
	@echo
	@echo "make align_seq	Build only the sequential version"
	@echo "make align_omp	Build only the OpenMP version"
	@echo "make align_mpi	Build only the MPI version"
	@echo "make align_cuda	Build only the CUDA version"
	@echo
	@echo "make all	Build all versions (Sequential, OpenMP, MPI, CUDA)"
	@echo "make debug	Build all version with demo output for small sequences"
	@echo "make clean	Remove targets"
	@echo

all: $(OBJS)

align_seq: align.c rng.c
	$(CC) $(FLAGS) $(DEBUG) $< $(LIBS) -o $@

align_omp: align_omp.c rng.c
	$(CC) $(FLAGS) $(DEBUG) $(OMPFLAG) $< $(LIBS) -o $@

align_mpi: align_mpi.c rng.c
	$(MPICC) $(FLAGS) $(DEBUG) $< $(LIBS) -o $@

align_cuda: align_cuda.cu rng.c
	$(CUDACC) $(CUDAFLAGS) $(DEBUG) $< $(LIBS) -o $@


# Remove the target files
clean:
	rm -rf $(OBJS)

# Compile in debug mode
debug:
	make DEBUG="-DDEBUG -g" all
