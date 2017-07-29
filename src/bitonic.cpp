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

void print(uint* host_data, uint n) {
	std::cout << "\n";
	for (uint i = 0; i < n; i++) {
		std::cout << host_data[i] << " ";
	}
	std::cout << "\n";
}

void swap(bool up, uint* a, int p, int q) {
	if ((a[p] > a[q]) == up) {
		uint aux = a[p];
		a[p] = a[q];
		a[q] = aux;
	}
}

/*void merge(uint* a, int n, int c){
 int p = 1;
 for(int j = c; j >= 2; j/=2){
 int m = j/2;

 for(int i = 0; i < n/j; i++){
 for(int k = i*j; k < (i*j+m); k++) {
 swap((i&p)==0, a, k, m+k);
 }
 }

 p*=2;
 }
 }*/

void bitonicSort(int logn, uint* a, int n) {

	/*for(int k = 1; k <= n; k*=2) {
	 merge(a, n, k);
	 }*/

	for (int c = 0; c <= logn; c++) {

		int x = 1 << c;

		int p = 1;

		for (int j = x; j >= 2; j /= 2) {

			int m = j >> 1;

			for (int i = 0; i < n / j; i++) {

				for (int k = i * j; k < (i * j + m); k++) {

					swap((i & p) == 0, a, k, m + k);
				}
			}

			p = p << 1;
		}
	}
}

int main(int argc, char **argv) {
	uint num_of_elements;
	uint i;

	scanf("%d", &num_of_elements);
	int n = num_of_elements * num_of_elements;
	uint mem_size = sizeof(int) * n;
	uint *h_vec = (uint *) malloc(mem_size);
	for (int i = 0; i < num_of_elements; i++) {
		for (int j = 0; j < num_of_elements; j++) {
			scanf("%d", &h_vec[i * num_of_elements + j]);
		}
	}

	int logn = 0;
	for(int i = 1; i < n; i*=2){
		logn++;
	}

	printf("logn=%d\n", logn);

	print(h_vec, n);

	bitonicSort(logn, h_vec, n);

	print(h_vec, n);

	return 0;
}
