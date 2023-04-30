// %%cu Google Colap 에서 실행할 때 

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define SIZE_OF_ARR 100

void CUDA_ERR_CHECK(cudaError_t err, int lineNo)
{
	if(err != cudaSuccess)
	{
		printf("Cuda Err : %s , Check Line No : %d\n",cudaGetErrorString(err),lineNo);
	}
}

__global__ void	MultipleFunctionInGPU(int *DEVICE_arrA,int *DEVICE_arrB,int *DEVICE_arrC)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < SIZE_OF_ARR)
	{
    DEVICE_arrC[i] = DEVICE_arrA[i] + DEVICE_arrB[i];
    //atomicAdd(&DEVICE_arrC[i],DEVICE_arrA[i] + DEVICE_arrB[i]);
	}
}

int main()
{
	int *HOST_arrA, *HOST_arrB, *HOST_arrC;	// CPU에서 사용할 메모리
	int *DEVICE_arrA, *DEVICE_arrB, *DEVICE_arrC; // GPU에서 사용할 메모리

	cudaError_t err;

	HOST_arrA = (int *)malloc(SIZE_OF_ARR  * sizeof(int));         
	HOST_arrB = (int *)malloc(SIZE_OF_ARR  * sizeof(int));				
	HOST_arrC = (int *)malloc(SIZE_OF_ARR  * sizeof(int));				 

	// 변수 초기화
	for (int i = 0; i < SIZE_OF_ARR; i++)
	{
			HOST_arrB[i] = 1;
			HOST_arrA[i] = 2;
			HOST_arrC[i] = 0;
	}

	// GPU Memory에 할당하는 변수
	err = cudaMalloc(&DEVICE_arrA, SIZE_OF_ARR * sizeof(int));	CUDA_ERR_CHECK(err,__LINE__) ;
	err = cudaMemcpy(DEVICE_arrA ,HOST_arrA ,SIZE_OF_ARR  * sizeof(int) , cudaMemcpyHostToDevice);	CUDA_ERR_CHECK(err,__LINE__) ;

	err = cudaMalloc(&DEVICE_arrB, SIZE_OF_ARR * sizeof(int));	CUDA_ERR_CHECK(err,__LINE__) ;
	err = cudaMemcpy(DEVICE_arrB ,HOST_arrB ,SIZE_OF_ARR  * sizeof(int) , cudaMemcpyHostToDevice);	CUDA_ERR_CHECK(err,__LINE__) ;
	
	err = cudaMalloc(&DEVICE_arrC, SIZE_OF_ARR * sizeof(int));	CUDA_ERR_CHECK(err,__LINE__) ;
	err = cudaMemcpy(DEVICE_arrC ,HOST_arrC ,SIZE_OF_ARR  * sizeof(int) , cudaMemcpyHostToDevice);	CUDA_ERR_CHECK(err,__LINE__) ;

	MultipleFunctionInGPU<<<10,10>>>(DEVICE_arrA, DEVICE_arrB, DEVICE_arrC); 
	cudaDeviceSynchronize();

	cudaMemcpy(HOST_arrC ,DEVICE_arrC ,SIZE_OF_ARR * sizeof(int) , cudaMemcpyDeviceToHost);

	free(HOST_arrA);	cudaFree( DEVICE_arrA);
	free(HOST_arrB);	cudaFree( DEVICE_arrB);
	free(HOST_arrC);	cudaFree( DEVICE_arrC);

	cudaDeviceReset();

	
}