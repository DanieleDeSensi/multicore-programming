CC          = gcc
LIBS        =
CFLAGS      = -Wall -pedantic -g
OMPFLAGS    = -fopenmp

RM=rm -f

EXES=histogram histogram_solution_trivial_crit histogram_solution_trivial_ato histogram_solution_ato_local histogram_solution_red histogram_solution_ato_local_better

all: $(EXES)

%: %.c 
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $@.c $(LIBS)

clean:
	$(RM) $(EXES) 