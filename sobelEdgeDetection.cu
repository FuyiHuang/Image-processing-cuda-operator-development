#include <iostream>
#include <opencv2/opencv.hpp>


__global__ void sobelEdgeDetection(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Each thread computes one pixel
    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        // Gradient x and y
        float Gx = 0;
        float Gy = 0;

        // Apply the Sobel operator
        for(int dx = -1; dx <= 1; dx++) {
            for(int dy = -1; dy <= 1; dy++) {
                int pixel = input[(y + dy) * width + (x + dx)];
                Gx += pixel * dx;
                Gy += pixel * dy;
            }
        }

        // Compute the resulting intensity
        int intensity = sqrtf(Gx * Gx + Gy * Gy);

        // Write the resulting intensity to the output image
        output[y * width + x] = intensity;
    }
}


int main() {
    // Load an image using OpenCV
    cv::Mat img = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);

    // Create device variables
    unsigned char* d_input;
    unsigned char* d_output;

    // Allocate device memory
    cudaMalloc(&d_input, img.total());
    cudaMalloc(&d_output, img.total());

    // Copy data from host to device
    cudaMemcpy(d_input, img.data, img.total(), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 block(32, 32);
    dim3 grid((img.cols + block.x - 1) / block.x, (img.rows + block.y - 1) / block.y);

    // Call the kernel function
    sobelEdgeDetection<<<grid, block>>>(d_input, d_output, img.cols, img.rows);

    // Copy data from device to host
    cudaMemcpy(img.data, d_output, img.total(), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Display the output image
    cv::imshow("Edge Image", img);
    cv::waitKey(0);

    return 0;
}