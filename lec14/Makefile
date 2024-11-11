CC          = gcc
LIBS        =
CFLAGS      = -fopenmp -Wall -pedantic -g

RM=rm -f

EXES=pi pi_solution_crit_rand pi_solution_crit pi_solution_ato pi_solution_red matmul matmul_solution histogram histogram_solution_trivial_crit histogram_solution_trivial_ato histogram_solution_ato_local histogram_solution_red histogram_solution_ato_local_better

all: $(EXES)

%: %.c 
	$(CC) $(CFLAGS) -o $@ $@.c $(LIBS)

clean:
	$(RM) $(EXES) 