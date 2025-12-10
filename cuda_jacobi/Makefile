CC          = nvcc
LIBS        =
CFLAGS      = -arch=compute_80 -DBLUR_SIZE=1 -DBLOCK_SIZE=16

RM=rm -f

EXES=jacobi jacobi_solution

all: $(EXES)

%: %.cu 
	$(CC) $(CFLAGS) -o $@ $@.cu $(LIBS)

clean:
	$(RM) $(EXES) 
