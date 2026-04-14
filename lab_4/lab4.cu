#include <iostream>
#include <vector>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#define CSC(call) { \
    cudaError_t res = call; \
    if (res != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(res)); \
        exit(0); \
    } \
}

struct abs_comparator {
    double* data;
    int n2;
    int k;
    __device__ bool operator()(int i, int j) const {
        return fabs(data[i * n2 + k]) < fabs(data[j * n2 + k]);
    }
};

__global__ void swap_rows_kernel(double* matrix, int n2, int row1, int row2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        double temp = matrix[row1 * n2 + idx];
        matrix[row1 * n2 + idx] = matrix[row2 * n2 + idx];
        matrix[row2 * n2 + idx] = temp;
    }
}

// Передаем divisor как значение, чтобы избежать race condition
__global__ void normalize_kernel(double* matrix, int n2, int k, double divisor) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n2) {
        matrix[k * n2 + col] /= divisor;
    }
}

// Исключаем влияние race condition: не трогаем k-й столбец
__global__ void eliminate_kernel(double* matrix, int n, int n2, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n2 && row != k) {
        // Все потоки в строке 'row' читают один и тот же 'factor'
        // Он не меняется, пока col != k
        if (col != k) {
            double factor = matrix[row * n2 + k];
            matrix[row * n2 + col] -= factor * matrix[k * n2 + col];
        }
    }
}

int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    if (n == 0) return 0;

    int n2 = 2 * n;
    std::vector<double> h_matrix(n * n2, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%lf", &h_matrix[i * n2 + j]);
        }
        h_matrix[i * n2 + (n + i)] = 1.0;
    }

    double* d_matrix;
    CSC(cudaMalloc(&d_matrix, sizeof(double) * n * n2));
    CSC(cudaMemcpy(d_matrix, h_matrix.data(), sizeof(double) * n * n2, cudaMemcpyHostToDevice));

    thrust::counting_iterator<int> iter_begin(0);
    thrust::counting_iterator<int> iter_end(n);

    for (int k = 0; k < n; k++) {
        // 1. Поиск максимума (Pivoting)
        abs_comparator comp = {d_matrix, n2, k};
        auto pivot_iter = thrust::max_element(thrust::device, iter_begin + k, iter_end, comp);
        int pivot_row = *pivot_iter;

        // 2. Перестановка строк
        if (pivot_row != k) {
            swap_rows_kernel<<<(n2 + 255) / 256, 256>>>(d_matrix, n2, k, pivot_row);
        }

        // 3. Получаем делитель на CPU, чтобы передать в ядро как константу
        double h_pivot_val;
        CSC(cudaMemcpy(&h_pivot_val, &d_matrix[k * n2 + k], sizeof(double), cudaMemcpyDeviceToHost));

        // 4. Нормализация
        normalize_kernel<<<(n2 + 255) / 256, 256>>>(d_matrix, n2, k, h_pivot_val);

        // 5. Исключение (2D сетка)
        dim3 threads(32, 32);
        dim3 blocks((n2 + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);
        eliminate_kernel<<<blocks, threads>>>(d_matrix, n, n2, k);
        
        CSC(cudaDeviceSynchronize());
    }

    CSC(cudaMemcpy(h_matrix.data(), d_matrix, sizeof(double) * n * n2, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.10e%c", h_matrix[i * n2 + (n + j)], (j == n - 1) ? '\n' : ' ');
        }
    }

    CSC(cudaFree(d_matrix));
    return 0;
}
