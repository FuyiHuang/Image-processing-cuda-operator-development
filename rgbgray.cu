#include <opencv2/opencv.hpp>


__global__ void rgbToGrayscaleKernel(unsigned char* grayscaleImage, unsigned char* rgbImage, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Check if the thread is inside the image bounds
    if (x < width && y < height) {
        // Calculate the grayscale value using the NTSC conversion formula
        int index = y * width + x;
        float grayscaleValue = 0.299f * rgbImage[3 * index] 
                             + 0.587f * rgbImage[3 * index + 1] 
                             + 0.114f * rgbImage[3 * index + 2];

        grayscaleImage[index] = static_cast<unsigned char>(grayscaleValue);
    }
}

int main()
{
    /// Load the RGB image using OpenCV
    cv::Mat img = cv::imread("image.jpg");

    // Check if the image has been loaded correctly
    if (img.empty()) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    // Allocate host memory for the grayscale image
    unsigned char* h_grayscaleImage = new unsigned char[img.total()];

    // Get the pointer to the RGB image data
    unsigned char* h_rgbImage = img.data;

    // Allocate memory on the GPU
    unsigned char* d_rgbImage;
    unsigned char* d_grayscaleImage;
    int width = 256;
    int height = 256;
    size_t numPixels = width * height;
    cudaMalloc(&d_rgbImage, numPixels * 3 * sizeof(unsigned char));
    cudaMalloc(&d_grayscaleImage, numPixels * sizeof(unsigned char));

    // Copy the RGB image data to the GPU
    cudaMemcpy(d_rgbImage, h_rgbImage, numPixels * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Specify a reasonable block and grid size
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Convert the image to grayscale
    rgbToGrayscaleKernel<<<gridSize, blockSize>>>(d_grayscaleImage, d_rgbImage, width, height);

    // Wait for all threads to finish
    cudaDeviceSynchronize();

    // Copy the grayscale image data back to the CPU
    cudaMemcpy(h_grayscaleImage, d_grayscaleImage, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // After calling cudaMemcpy to copy the grayscale image data back to the CPU
    // Create an OpenCV Mat for the grayscale image
    cv::Mat grayImg(img.rows, img.cols, CV_8U, h_grayscaleImage);

    // Display the grayscale image
    cv::imshow("Grayscale Image", grayImg);
    cv::waitKey(0);

    // Delete the grayscale image data
    delete[] h_grayscaleImage;

    return 0;

    // Clean up
    cudaFree(d_rgbImage);
    cudaFree(d_grayscaleImage);

    return 0;
}
