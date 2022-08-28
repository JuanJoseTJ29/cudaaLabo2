#include <thread>
#include <time.h>
#include <iostream>
#include <math.h>
#include "imageLoader.cpp"

#define GRIDVAL 20.0

__global__ void sobel_gpu(const byte *orig, byte *cpu, const unsigned int width, const unsigned int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float dx, dy;
    if (x > 0 && y > 0 && x < width - 1 && y < height - 1)
    {
        dx = (-1 * orig[(y - 1) * width + (x - 1)]) + (-2 * orig[y * width + (x - 1)]) + (-1 * orig[(y + 1) * width + (x - 1)]) +
             (orig[(y - 1) * width + (x + 1)]) + (2 * orig[y * width + (x + 1)]) + (orig[(y + 1) * width + (x + 1)]);
        dy = (orig[(y - 1) * width + (x - 1)]) + (2 * orig[(y - 1) * width + x]) + (orig[(y - 1) * width + (x + 1)]) +
             (-1 * orig[(y + 1) * width + (x - 1)]) + (-2 * orig[(y + 1) * width + x]) + (-1 * orig[(y + 1) * width + (x + 1)]);
        cpu[y * width + x] = sqrt((dx * dx) + (dy * dy));
    }
}

int main(int argc, char *argv[])
{

    imgData origImg = loadImage(argv[1]);
    imgData gpuImg(new byte[origImg.width * origImg.height], origImg.width, origImg.height);

    byte *gpu_orig, *gpu_sobel;
    cudaMalloc((void **)&gpu_orig, (origImg.width * origImg.height));
    cudaMalloc((void **)&gpu_sobel, (origImg.width * origImg.height));

    cudaMemcpy(gpu_orig, origImg.pixels, (origImg.width * origImg.height), cudaMemcpyHostToDevice);
    cudaMemset(gpu_sobel, 0, (origImg.width * origImg.height));

    dim3 threadsPerBlock(GRIDVAL, GRIDVAL, 1);
    dim3 numBlocks(ceil(origImg.width / GRIDVAL), ceil(origImg.height / GRIDVAL), 1);

    sobel_gpu<<<numBlocks, threadsPerBlock>>>(gpu_orig, gpu_sobel, origImg.width, origImg.height);
    cudaError_t cudaerror = cudaDeviceSynchronize();

    if (cudaerror != cudaSuccess)
        fprintf(stderr, "Cuda failed to synchronize: %s\n", cudaGetErrorName(cudaerror));

    cudaMemcpy(gpuImg.pixels, gpu_sobel, (origImg.width * origImg.height), cudaMemcpyDeviceToHost);

    writeImage(argv[1], "gpu", gpuImg);

    cudaFree(gpu_orig);
    cudaFree(gpu_sobel);
    return 0;
}
