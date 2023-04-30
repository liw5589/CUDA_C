// %%cu Google Colap 에서 실행할 때 

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define SIZE_OF_ARR 3

void  CUDA_ERR_CHECK(cudaError_t err, int lineNo)
{
    if (err != cudaSuccess)
    {
        printf("Cuda Err : %s , Check Line No : %d\n", cudaGetErrorString(err), lineNo);
    }
}

__global__ void MultipleFunctionInGPU(int *DEVICE_arrA, int *DEVICE_arrB, int *DEVICE_arrC)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < SIZE_OF_ARR && j < SIZE_OF_ARR)
    {
        // DEVICE_arrC[i] = DEVICE_arrA[i * SIZE_OF_ARR + j] + DEVICE_arrB[j];
        atomicAdd(&DEVICE_arrC[i], DEVICE_arrA[i * SIZE_OF_ARR + j] * DEVICE_arrB[j]);
    }
}

int main()
{
    int *HOST_arrA, *HOST_arrB, *HOST_arrC;       // CPU에서 사용할 메모리
    int *DEVICE_arrA, *DEVICE_arrB, *DEVICE_arrC; // GPU에서 사용할 메모리

    cudaError_t err;

    HOST_arrA = (int *)malloc(SIZE_OF_ARR * SIZE_OF_ARR * sizeof(int)); // 1000 * 1000
    HOST_arrB = (int *)malloc(SIZE_OF_ARR * sizeof(int));               // 1000 * 1
    HOST_arrC = (int *)malloc(SIZE_OF_ARR * sizeof(int));               // 1000 * 1

    // 변수 초기화
    for (int i = 0; i < SIZE_OF_ARR; i++)
    {
        HOST_arrB[i] = i + 1;
        for (int j = 0; j < SIZE_OF_ARR; j++)
        {
            HOST_arrA[i * SIZE_OF_ARR + j] = 1;
        }
    }

    // GPU Memory에 할당하는 변수
    err = cudaMalloc(&DEVICE_arrA, SIZE_OF_ARR * SIZE_OF_ARR * sizeof(int));
    CUDA_ERR_CHECK(err, __LINE__);
    err = cudaMemcpy(DEVICE_arrA, HOST_arrA, SIZE_OF_ARR * SIZE_OF_ARR * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK(err, __LINE__);

    err = cudaMalloc(&DEVICE_arrB, SIZE_OF_ARR * sizeof(int));
    CUDA_ERR_CHECK(err, __LINE__);
    err = cudaMemcpy(DEVICE_arrB, HOST_arrB, SIZE_OF_ARR * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK(err, __LINE__);

    err = cudaMalloc(&DEVICE_arrC, SIZE_OF_ARR * sizeof(int));
    CUDA_ERR_CHECK(err, __LINE__);
    err = cudaMemcpy(DEVICE_arrC, HOST_arrC, SIZE_OF_ARR * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK(err, __LINE__);

    dim3 DimGrid(3, 3);
    dim3 DimBlock(SIZE_OF_ARR / 3, SIZE_OF_ARR / 3);

    MultipleFunctionInGPU<<<DimGrid, DimBlock>>>(DEVICE_arrA, DEVICE_arrB, DEVICE_arrC);
    cudaDeviceSynchronize();

    cudaMemcpy(HOST_arrC, DEVICE_arrC, SIZE_OF_ARR * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < SIZE_OF_ARR; i++)
    {
        printf("Host C [%d] : %d \n", i, HOST_arrC[i]);
    }

    free(HOST_arrA);
    cudaFree(DEVICE_arrA);
    free(HOST_arrB);
    cudaFree(DEVICE_arrB);
    free(HOST_arrC);
    cudaFree(DEVICE_arrC);

    cudaDeviceReset();
}