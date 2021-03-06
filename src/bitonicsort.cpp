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

void kernel(uint* a, int n, int p, int q) {
	int d = 1 << (p-q);

	for(int i = 0; i < n; i++) {
		bool up = ((i >> p) & 2) == 0;

		if ((i & d) == 0 && (a[i] > a[i | d]) == up) {
			int t = a[i];
			a[i] = a[i | d];
			a[i | d] = t;
		}
	}
}

void bitonicSort(int logn, uint* a, int n) {

	for (int p = 0; p < logn; p++) {
		for (int q = 0; q <= p; q++) {
			kernel(a, n, p, q);
		}
	}
}

void print(uint* host_data, uint n) {
	std::cout << "\n";
	for (uint i = 0; i < n; i++) {
		std::cout << host_data[i] << " ";
	}
	std::cout << "\n";
}

int main(int argc, char **argv) {
	uint num_of_elements;
	uint i;

	scanf("%d", &num_of_elements);
	int n = num_of_elements * num_of_elements;
	uint mem_size = sizeof(int) * (n);
	uint *h_vec = (uint *) malloc(mem_size);
	for (int i = 0; i < num_of_elements; i++) {
		for (int j = 0; j < num_of_elements; j++) {
			scanf("%d", &h_vec[i*num_of_elements + j]);
		}
	}

	int logn = 0;
	for(int i = 1; i < n; i*=2){
		logn++;
	}

	printf("logn=%d\n", logn);

	bitonicSort(logn, h_vec, n);

	print(h_vec, n);

	return 0;
}
