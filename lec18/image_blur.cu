#include <cuda.h>
#include <stdio.h>
#include "../lec13/my_timer.h"

#define HEIGHT 10240
#define WIDTH 10240
#define NUM_PIXELS (HEIGHT*WIDTH)
#define NUM_CHANNELS 1
#define BLUR_SIZE 1

__global__ void blurKernel(unsigned char* in, unsigned char* out, int w, int h){
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	if(col < w && row < h){
		int pixVal = 0;
		int pixels = 0;
		for(int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow){
			for(int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol){
				int curRow = row + blurRow;
				int curCol = col + blurCol;
				if(curRow >= 0 && curRow <= h && curCol >= 0 && curCol <= w){
					pixVal += in[curRow*w + curCol];
					pixels++;
				}
			}

		}
	        out[row*w + col] = (unsigned char) (pixVal / pixels);
	}
}

int main(int argc, char** argv){
	// We do not actually load an image
	// In principle, we should load an image from a file into a host buffer
	// and then copy it to a device buffer.
	// Instead, we only allocate input/output device buffers.
	size_t numBytes = NUM_PIXELS*NUM_CHANNELS*sizeof(unsigned char);
	unsigned char *d_input, *d_output;
	cudaMalloc((void**) &d_input, numBytes);
	cudaMalloc((void**) &d_output, numBytes);
	
	dim3 dimGrid(ceil(WIDTH/16.0), ceil(HEIGHT/16.0));
	dim3 dimBlock(16, 16);
	double start, stop;

	GET_TIME(start);
	blurKernel<<<dimGrid, dimBlock>>>(d_input, d_output, WIDTH, HEIGHT);
	cudaDeviceSynchronize();
	GET_TIME(stop);

	printf("Runtime: %lf seconds\n", stop - start);

	cudaFree(d_input);
	cudaFree(d_output);
	return 0;
}
