#include <cuda.h>
#include <assert.h>
#include <stdio.h>
#include "../lec13/my_timer.h"

#define HEIGHT 8192
#define WIDTH 8192
#define NUM_PIXELS (HEIGHT*WIDTH)
#define NUM_CHANNELS 1

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
				if(curRow >= 0 && curRow < h && curCol >= 0 && curCol < w){
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
	// Instead, we create an 'image' made of random bytes.

	size_t numBytes = NUM_PIXELS*NUM_CHANNELS*sizeof(unsigned char);

	// Allocate host input/output vectors
	unsigned char *h_input, *h_output, *h_output_ref;
	h_input = (unsigned char*) malloc(numBytes);
	h_output = (unsigned char*) malloc(numBytes);
	h_output_ref = (unsigned char*) malloc(numBytes);
	

	printf("Initializing input.\n"); fflush(stdout);
	// Initialize the input with random stuff
	for(size_t i = 0; i < NUM_PIXELS*NUM_CHANNELS; i++){
		   h_input[i] = rand() % 256;
	}
	printf("Input initialized.\n"); fflush(stdout);

	// Allocate device input/output vectors and copy the input data to the device
	unsigned char *d_input, *d_output;
	cudaMalloc((void**) &d_input, numBytes);
	cudaMalloc((void**) &d_output, numBytes);
	cudaMemcpy(d_input, h_input, numBytes, cudaMemcpyHostToDevice);

	printf("Data copied to device.\n"); fflush(stdout);
	
	dim3 dimGrid(ceil(WIDTH/(float) BLOCK_SIZE), ceil(HEIGHT/ (float) BLOCK_SIZE));
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	double start, stop;

	// In principle we should also include in the timing the time to copy data to/from the device,
	// we do not do it here mostly to show the difference in time between the version with/without shared memory
	GET_TIME(start);
	blurKernel<<<dimGrid, dimBlock>>>(d_input, d_output, WIDTH, HEIGHT);
	cudaDeviceSynchronize();
	GET_TIME(stop);
	printf("Runtime: %lf seconds\n", stop - start); fflush(stdout);

	// Copy the output data from the device to the host
	cudaMemcpy(h_output, d_output, numBytes, cudaMemcpyDeviceToHost);
	printf("Output retrieved.\n"); fflush(stdout);

	// Now we check that the result computed by the device is correct
	// Do the same blurring on the host
	for(int row = 0; row < HEIGHT; row++){
		for(int col = 0; col < WIDTH; col++){
			int pixVal = 0;
	                int pixels = 0;
        	    	for(int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow){
				for(int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol){
                                    	    int curRow = row + blurRow;
                                	    int curCol = col + blurCol;
                                	    if(curRow >= 0 && curRow < HEIGHT && curCol >= 0 && curCol < WIDTH){
                                            	      pixVal += h_input[curRow*WIDTH + curCol];
                                        	      pixels++;
                                	    }
                        	}
			}
                	h_output_ref[row*WIDTH + col] = (unsigned char) (pixVal / pixels);
		}
	}
	printf("Reference result computed\n"); fflush(stdout);
	// Now check that the content of h_output is equal to h_output_ref
        for(size_t i = 0; i < NUM_PIXELS*NUM_CHANNELS; i++){
		   if(h_output_ref[i] != h_output[i]){
		   		  fprintf(stderr, "Outputs differ at index %d (%d vs. %d)\n", i, h_output_ref[i], h_output[i]); fflush(stderr);
				  exit(-1);
		   }

	}

	printf("Everything is fine\n"); fflush(stdout);

	cudaFree(d_input);
	cudaFree(d_output);
	return 0;
}
