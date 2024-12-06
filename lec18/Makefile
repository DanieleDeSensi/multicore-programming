CC          = nvcc
LIBS        =
CFLAGS      = -arch=sm_75

RM=rm -f

EXES=vector_add test_cuda image_blur

all: $(EXES)

%: %.cu 
	$(CC) $(CFLAGS) -o $@ $@.cu $(LIBS)

clean:
	$(RM) $(EXES) 
