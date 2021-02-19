#include <iostream>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
// Add elements of two arrays
__global__ void add(int n, float* x, float* y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	y[index] = x[index] + y[index];
}

// Multiply two matrices
__global__ void matrix_mult(int n, float* x, float* y, float* c) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int res = 0;
	if (row < n && column < n) {
		for (int i = 0; i < n; i++)
			res += x[row * n + i] * y[i * n + column];
	}
	c[(row * n) + column] = res;
}

void matrix_main() {
	// N*N matrix
	int N = 30;
	float* x, * y, * c;

	cudaMallocManaged(&x, N * N);
	cudaMallocManaged(&y, N * N);
	cudaMallocManaged(&c, N * N);

	// Randomize matrices
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			x[i * N + j] = rand() % 101;
			y[i * N + j] = rand() % 101;
		}
	}

	dim3 threadsPerBlock(N, N);
	matrix_mult<<<1, threadsPerBlock>>>(N, x, y, c);
	cudaDeviceSynchronize();

	// Matrix multiply on CPU
	float* correctC = new float[N * N];
	float res;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			res = 0;
			for (int k = 0; k < N; k++) {
				res += x[i * N + k] * y[k * N + j];
			}
			correctC[i * N + j] = res;
		}
	}
	// Check for max errors
	double maxError = 0;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			maxError = fmax(maxError, fabs(c[(i * N) + j] - correctC[(i * N) + j]));
		}
	}
	std::cout << "Max error: " << maxError << std::endl;
	cudaFree(x);
	cudaFree(y);
	cudaFree(c);
}

int main(void) {
	matrix_main();
	add_main();
}

void add_main()
{
	int N = 1 << 20; // 1M elements
	float* x, * y;
	cudaMallocManaged(&x, N * sizeof(float));
	cudaMallocManaged(&y, N * sizeof(float));
	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}
	// Run kernel on 1M elements on the CPU
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	printf("%d", numBlocks);
	add<<<numBlocks, blockSize>>>(N, x, y);
	cudaDeviceSynchronize();

	// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
		maxError = fmax(maxError, fabs(y[i] - 3.0f));
	std::cout << "Max error: " << maxError << std::endl;
	// Free memory
	cudaFree(x);
	cudaFree(y);
}