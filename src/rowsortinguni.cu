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

void print(uint* host_data, uint n, uint m) {
	std::cout << "\n";
	for (uint i = 0; i < n; i++) {
		for (uint j = 0; j < m; j++) {
			std::cout << host_data[i * n + j] << "\t";
		}
		std::cout << "\n";
	}

}

__global__ void bitonic_sort_step(uint *dev_values, int k, int p, int n) {

	uint i = blockDim.x * blockIdx.x + threadIdx.x;

	uint ixp = i ^ p;

	/* The threads with the lowest ids sort the array. */
	if (i < ixp) {

		bool up = ((i & k) == 0); // sorting the entire matrix row

		// Sort ascending or descending according to up value
		if ((dev_values[i] > dev_values[ixp]) == up) {
			// exchange(i,ixj);
			uint temp = dev_values[i];
			dev_values[i] = dev_values[ixp];
			dev_values[ixp] = temp;
		}

	}
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sorting(uint *values, int n, int m) {

	dim3 dimBlock(BLOCK_SIZE, 1);
	dim3 dimGrid((n*m - 1) / dimBlock.x + 1, 1);

	// sorting the entire matrix row
	for (int k = 2; k <= n; k <<= 1)
		for (int p = k >> 1; p > 0; p = p >> 1)
			bitonic_sort_step<<<dimGrid, dimBlock>>>(values, k, p, n);
}

int main(int argc, char** argv) {
	uint num_of_elements;
	scanf("%d", &num_of_elements);
	int n = num_of_elements;
	int m = num_of_elements;
	uint mem_size = sizeof(int) * (n * m);
	uint *h_vec = (uint *) malloc(mem_size);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			scanf("%d", &h_vec[i * n + j]);
		}
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	uint *d_vec;

	cudaTest(cudaMalloc((void **) &d_vec, mem_size));

	for (int i = 0; i < EXECUTIONS; i++) {

		cudaTest(cudaMemcpy(d_vec, h_vec, mem_size, cudaMemcpyHostToDevice));

		cudaEventRecord(start);
		bitonic_sorting(d_vec, n, m);
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

	cudaMemcpy(h_vec, d_vec, mem_size, cudaMemcpyDeviceToHost);

	cudaFree(d_vec);

	if (ELAPSED_TIME != 1) {
		print(h_vec, n, m);
	}

	free(h_vec);

	return 0;
}

/*
 * for (int p = 0; p < logn; p++) {
 for (int q = 0; q <= p; q++) {

 int d = 1 << (p-q);
 //for(int i = 0; i < n; i++) {
 bool up = ((col >> p) & 2) == 0;

 if ((col & d) == 0 && (As[row][col] > As[row][col | d]) == up) {
 int t = As[row][col];
 As[row][col] = As[row][col | d];
 As[row][col | d] = t;
 }
 //			}
 }
 }
 */
