CC          = gcc
LIBS        =
CFLAGS      = -Wall -pedantic -g
OMPFLAGS    = -fopenmp

RM=rm -f

EXES=pi pi_solution_crit_rand pi_solution_crit pi_solution_ato pi_solution_red matmul matmul_solution histogram histogram_solution_trivial_crit histogram_solution_trivial_ato histogram_solution_ato_local histogram_solution_red histogram_solution_ato_local_better rand_vs_rand_r

all: $(EXES)

rand_vs_rand_r: rand_vs_rand_r.c 
	$(CC) $(CFLAGS) -o $@ $@.c $(LIBS)

%: %.c 
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $@.c $(LIBS)

clean:
	$(RM) $(EXES) 