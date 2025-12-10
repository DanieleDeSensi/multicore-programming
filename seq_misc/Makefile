CC          = gcc
LIBS        =
CFLAGS      = -Wall -pedantic -g
OMPFLAGS    = -fopenmp

RM=rm -f

EXES=rand_vs_rand_r

all: $(EXES)

rand_vs_rand_r: rand_vs_rand_r.c 
	$(CC) $(CFLAGS) -o $@ $@.c $(LIBS)

%: %.c 
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $@.c $(LIBS)

clean:
	$(RM) $(EXES) 