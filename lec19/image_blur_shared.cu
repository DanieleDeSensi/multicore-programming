#include <cuda.h>
#include <stdio.h>
#include "../lec13/my_timer.h"

#define HEIGHT 10240
#define WIDTH 10240
#define NUM_PIXELS (HEIGHT*WIDTH)
#define NUM_CHANNELS 1
#define BLUR_SIZE 1
#define BLOCK_SIZE 16

__global__ void blurKernel(unsigned char* in, unsigned char* out, int w, int h){
        __device__ __shared__ unsigned char in_shared[BLOCK_SIZE*BLOCK_SIZE];
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	if(col < w && row < h){
		in_shared[threadIdx.y*BLOCK_SIZE + threadIdx.x] = in[row*w + col];
		__syncthreads();

		int pixVal = 0;
		int pixels = 0;
		for(int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow){
			for(int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol){
				int curRow = row + blurRow;
				int curCol = col + blurCol;
				if(curRow >= 0 && curRow <= h && curCol >= 0 && curCol <= w){					
					// If the pixel is out of the block, take it from global mem
					if(curRow / blockDim.y != row / blockDim.y ||
				           curCol / blockDim.x != col / blockDim.x){
						pixVal += in[curRow*w + curCol];
					}else{
						pixVal += in_shared[(curRow % BLOCK_SIZE)*BLOCK_SIZE + (curCol % BLOCK_SIZE)];
					}
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
	
	dim3 dimGrid(ceil(WIDTH/(float)BLOCK_SIZE), ceil(HEIGHT/(float)BLOCK_SIZE));
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
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
