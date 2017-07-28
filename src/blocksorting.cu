/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <cuda.h>

// Thread block size
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif


void cudaTest(cudaError_t error) {
	if (error != cudaSuccess) {
		printf("cuda returned error %s (code %d), line(%d)\n",
				cudaGetErrorString(error), error, __LINE__);
		exit (EXIT_FAILURE);
	}
}

void print(uint* host_data, uint n) {
	std::cout << "\n";
	for (uint i = 0; i < n; i++) {
		std::cout << host_data[i] << " ";
	}
	std::cout << "\n";
}

// Matrix multiplication kernel called by MatMul()
__global__ void block_sorting(uint* A, int n, int m) {
// Each thread computes one element of C
// by accumulating results into Cvalue
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;


	for (int e = 0; e < A.width; ++e)
		Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];

	C.elements[row * C.width + col] = Cvalue;

}


int main(int argc, char** argv) {

	uint num_of_elements;
	uint i;

	scanf("%d", &num_of_elements);
	uint mem_size = sizeof(int) * (num_of_elements * num_of_elements);
	uint *h_vec = (uint *) malloc(mem_size);
	for (i = 0; i < num_of_elements; i++) {
		for (i = 0; i < num_of_elements + 1; i++) {
			scanf("%d", &h_vec[i]);
		}
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	uint *d_vec;

	cudaTest(cudaMalloc((void **) &d_vec, mem_size_vec));

	for (int i = 0; i < EXECUTIONS; i++) {

		cudaTest(cudaMemcpy(d_vec, h_vec, mem_size_vec, cudaMemcpyHostToDevice));

		cudaEventRecord(start);
		block_sorting(d_vec, num_of_elements, num_of_elements);
		cudaEventRecord(stop);

		cudaError_t errSync = cudaGetLastError();
		cudaError_t errAsync = cudaDeviceSynchronize();
		if (errSync != cudaSuccess)
			printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
		if (errAsync != cudaSuccess)
			printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

		if (ELAPSED_TIME == 1) {
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			std::cout << milliseconds << "\n";
		}

		cudaDeviceSynchronize();
	}

	cudaMemcpy(h_vec, d_vec_out, mem_size_vec, cudaMemcpyDeviceToHost);

	cudaFree(d_vec);

	if (ELAPSED_TIME != 1) {
		print(h_vec, num_of_elements);
	}

	free(h_vec);

	return 0;
}

