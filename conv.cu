#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA kernel for image convolution
__global__ void imgConvolve(const float* input, float* output, int width, int height) {
    // Set up necessary indices and values for convolution
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    
    __shared__ float sharedMem[32][32];  // A block of shared memory for reducing global memory accesses

    // Load the necessary pixels into shared memory
    if (idx_x < width && idx_y < height) {
        sharedMem[threadIdx.y][threadIdx.x] = input[idx_y * width + idx_x];
    }
    __syncthreads();  // Ensure all threads have finished loading into shared memory

    // Perform convolution operation
    if (idx_x > 0 && idx_x < width - 1 && idx_y > 0 && idx_y < height - 1 && threadIdx.x > 0 && threadIdx.x < blockDim.x - 1 && threadIdx.y > 0 && threadIdx.y < blockDim.y - 1) {
        float result = 0.0f;
        for (int j = -1; j <= 1; j++) {
            #pragma unroll
            for (int i = -1; i <= 1; i++) {
                result += sharedMem[threadIdx.y + j][threadIdx.x + i];
            }
        }
        output[idx_y * width + idx_x] = result / 9.0f;
    }
}

// Main function
int main() {
    // Define image dimensions
    int imgWidth = 512;
    int imgHeight = 512;
    float h_input[imgHeight][imgWidth];
    float h_output[imgHeight][imgWidth];

    // Initialize image with random values
    for (int y = 0; y < imgHeight; y++) {
        for (int x = 0; x < imgWidth; x++) {
            h_input[y][x] = rand() / static_cast<float>(RAND_MAX);
        }
    }

    // Allocate device memory
    float* d_input;
    float* d_output;
    cudaMalloc((void**)&d_input, imgWidth * imgHeight * sizeof(float));
    cudaMalloc((void**)&d_output, imgWidth * imgHeight * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input, imgWidth * imgHeight * sizeof(float), cudaMemcpyHostToDevice);

    // Run convolution kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(imgWidth / threadsPerBlock.x, imgHeight / threadsPerBlock.y);
    imgConvolve<<<numBlocks, threadsPerBlock>>>(d_input, d_output, imgWidth, imgHeight);
    
    // Copy output to host
    cudaMemcpy(h_output, d_output, imgWidth * imgHeight * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Print first 5 values of the output for verification
    for (int i = 0; i < 5; i++) {
        printf("%f ", h_output[i]);
    }
    printf("\n");

    return 0;
}
