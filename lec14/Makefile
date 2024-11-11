CC          = gcc
CLINKER     = $(CC)
OPTFLAGS    = -fopenmp
LIBS        =

CFLAGS	  = $(OPTFLAGS) 

RM=rm -f

EXES=pi pi_solution_crit_rand pi_solution_crit pi_solution_ato pi_solution_red 

all: $(EXES)

pi: pi.c 
	$(CLINKER) $(OPTFLAGS) -o pi pi.c $(LIBS)

pi_solution_crit_rand: pi_solution_crit_rand.c
	$(CLINKER) $(OPTFLAGS) -o pi_solution_crit_rand pi_solution_crit_rand.c $(LIBS)

pi_solution_crit: pi_solution_crit.c
	$(CLINKER) $(OPTFLAGS) -o pi_solution_crit pi_solution_crit.c $(LIBS)

pi_solution_ato: pi_solution_ato.c
	$(CLINKER) $(OPTFLAGS) -o pi_solution_ato pi_solution_ato.c $(LIBS)

pi_solution_red: pi_solution_red.c
	$(CLINKER) $(OPTFLAGS) -o pi_solution_red pi_solution_red.c $(LIBS)

clean:
	$(RM) $(EXES) 