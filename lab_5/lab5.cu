#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#define CSC(call) { \
    cudaError_t res = call; \
    if (res != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(res)); \
        exit(0); \
    } \
}

__global__ void histogram_kernel(int* data, int* hist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while (idx < n) {
        atomicAdd(&hist[data[idx]], 1);
        idx += offset;
    }
}

__global__ void fill_kernel(int* offsets, int* output, int num_buckets, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_buckets) {
        int start = offsets[v];
        int end = (v == num_buckets - 1) ? n : offsets[v + 1];
        for (int i = start; i < end; ++i) {
            output[i] = v;
        }
    }
}

int main() {
    int n;
    if (fread(&n, sizeof(int), 1, stdin) != 1) return 0;
    if (n <= 0) return 0;

    int* h_data = (int*)malloc(sizeof(int) * n);
    fread(h_data, sizeof(int), n, stdin);

    int* d_data;
    CSC(cudaMalloc(&d_data, sizeof(int) * n));
    CSC(cudaMemcpy(d_data, h_data, sizeof(int) * n, cudaMemcpyHostToDevice));

    const int num_buckets = 1 << 24; 
    int* d_hist;
    CSC(cudaMalloc(&d_hist, sizeof(int) * num_buckets));
    CSC(cudaMemset(d_hist, 0, sizeof(int) * num_buckets));

    histogram_kernel<<<1024, 1024>>>(d_data, d_hist, n);

    thrust::device_ptr<int> d_ptr(d_hist);
    thrust::exclusive_scan(d_ptr, d_ptr + num_buckets, d_ptr);

    int* d_out;
    CSC(cudaMalloc(&d_out, sizeof(int) * n));
    
    fill_kernel<<<(num_buckets + 255) / 256, 256>>>(d_hist, d_out, num_buckets, n);

    CSC(cudaMemcpy(h_data, d_out, sizeof(int) * n, cudaMemcpyDeviceToHost));

    fwrite(h_data, sizeof(int), n, stdout);

    CSC(cudaFree(d_data));
    CSC(cudaFree(d_hist));
    CSC(cudaFree(d_out));
    free(h_data);
    return 0;
}
