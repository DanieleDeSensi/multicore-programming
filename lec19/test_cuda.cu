#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);

    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Number of CUDA devices: %d\n", count);
    return 0;
}

