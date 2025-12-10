CC          = gcc
LIBS        =
CFLAGS      = -Wall -pedantic -g
OMPFLAGS    = -fopenmp

RM=rm -f

EXES=matmul matmul_solution

all: $(EXES)

%: %.c 
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $@.c $(LIBS)

clean:
	$(RM) $(EXES) 