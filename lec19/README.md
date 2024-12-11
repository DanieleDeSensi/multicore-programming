- vector_add.cu: Example on vectors addition
- image_blur.cu: Example on image blurring with data in global memory
- image_blur_shared.cu: Example on image blurring with (part of the) data in shared memory
- jacobi.cu: Example on Jacobi solver (taken from https://github.com/csc-training/CUDA/tree/master/exercises/jacobi).
             It computes it both on CPU and GPU, comparing the runtime and the result.
	     This file only contains the CPU implementation, you are supposed to implement the GPU part.
- jacobi_solution.cu: Solution of jacobi.cu