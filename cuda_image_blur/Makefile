CC          = nvcc
LIBS        =
CFLAGS      = -arch=compute_80 -DBLUR_SIZE=1 -DBLOCK_SIZE=16

RM=rm -f

EXES=image_blur image_blur_shared image_blur_shared_all

all: $(EXES)

%: %.cu 
	$(CC) $(CFLAGS) -o $@ $@.cu $(LIBS)

clean:
	$(RM) $(EXES) 
