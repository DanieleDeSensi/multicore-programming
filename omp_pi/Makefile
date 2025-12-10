CC          = gcc
LIBS        =
CFLAGS      = -Wall -pedantic -g
OMPFLAGS    = -fopenmp

RM=rm -f

EXES=pi pi_solution_crit_rand pi_solution_crit pi_solution_ato pi_solution_red

all: $(EXES)

%: %.c 
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $@.c $(LIBS)

clean:
	$(RM) $(EXES) 