#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CSC(call) { \
    cudaError_t res = call; \
    if (res != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(res)); \
        exit(0); \
    } \
}

// Константная память для параметров классов
__constant__ double c_avg[32][3];
__constant__ double c_inv_cov[32][9];
__constant__ int c_nc;

__global__ void classify_kernel(uchar4* data, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t offset = (size_t)blockDim.x * gridDim.x;

    while (idx < n) {
        uchar4 p = data[idx];
        double p_vec[3] = {(double)p.x, (double)p.y, (double)p.z};

        double max_val = -1e18; 
        int best_class = 0;

        for (int j = 0; j < c_nc; j++) {
            double diff[3];
            diff[0] = p_vec[0] - c_avg[j][0];
            diff[1] = p_vec[1] - c_avg[j][1];
            diff[2] = p_vec[2] - c_avg[j][2];

            // Рассчитываем -(diff^T * inv_cov * diff)
            double current_val = 0;
            for (int r = 0; r < 3; r++) {
                double tmp = 0;
                for (int c = 0; c < 3; c++) {
                    tmp += diff[c] * c_inv_cov[j][r * 3 + c];
                }
                current_val += diff[r] * tmp;
            }
            current_val = -current_val;

            if (j == 0 || current_val > max_val) {
                max_val = current_val;
                best_class = j;
            }
        }
        data[idx].w = (unsigned char)best_class;
        idx += offset;
    }
}

// Инверсия матрицы 3x3 (CPU)
void invert3x3(double m[9], double inv[9]) {
    double det = m[0] * (m[4] * m[8] - m[5] * m[7]) -
                 m[1] * (m[3] * m[8] - m[5] * m[6]) +
                 m[2] * (m[3] * m[7] - m[4] * m[6]);
    double invDet = 1.0 / det;
    inv[0] = (m[4] * m[8] - m[5] * m[7]) * invDet;
    inv[1] = (m[2] * m[7] - m[1] * m[8]) * invDet;
    inv[2] = (m[1] * m[5] - m[2] * m[4]) * invDet;
    inv[3] = (m[5] * m[6] - m[3] * m[8]) * invDet;
    inv[4] = (m[0] * m[8] - m[2] * m[6]) * invDet;
    inv[5] = (m[2] * m[3] - m[0] * m[5]) * invDet;
    inv[6] = (m[3] * m[7] - m[4] * m[6]) * invDet;
    inv[7] = (m[1] * m[6] - m[0] * m[7]) * invDet;
    inv[8] = (m[0] * m[4] - m[1] * m[3]) * invDet;
}

int main() {
    char in_path[256], out_path[256];
    // Строго по PDF: первая строка - вход, вторая - выход
    if (scanf("%s", in_path) != 1) return 0;
    if (scanf("%s", out_path) != 1) return 0;

    int nc;
    if (scanf("%d", &nc) != 1) return 0;

    // Считываем выборки для каждого класса
    typedef struct { int x, y; } Point;
    Point** samples = (Point**)malloc(sizeof(Point*) * nc);
    int* sample_counts = (int*)malloc(sizeof(int) * nc);

    for (int i = 0; i < nc; i++) {
        scanf("%d", &sample_counts[i]);
        samples[i] = (Point*)malloc(sizeof(Point) * sample_counts[i]);
        for (int j = 0; j < sample_counts[i]; j++) {
            scanf("%d %d", &samples[i][j].x, &samples[i][j].y);
        }
    }

    // Читаем картинку
    FILE* fp = fopen(in_path, "rb");
    int w, h;
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);
    size_t n = (size_t)w * h;
    uchar4* h_data = (uchar4*)malloc(sizeof(uchar4) * n);
    fread(h_data, sizeof(uchar4), n, fp);
    fclose(fp);

    double h_avg[32][3];
    double h_inv_cov[32][9];

    for (int j = 0; j < nc; j++) {
        double sum[3] = {0, 0, 0};
        for (int i = 0; i < sample_counts[j]; i++) {
            uchar4 p = h_data[samples[j][i].y * w + samples[j][i].x];
            sum[0] += p.x; sum[1] += p.y; sum[2] += p.z;
        }
        h_avg[j][0] = sum[0] / sample_counts[j];
        h_avg[j][1] = sum[1] / sample_counts[j];
        h_avg[j][2] = sum[2] / sample_counts[j];

        double cov[9] = {0};
        for (int i = 0; i < sample_counts[j]; i++) {
            uchar4 p = h_data[samples[j][i].y * w + samples[j][i].x];
            double diff[3] = {p.x - h_avg[j][0], p.y - h_avg[j][1], p.z - h_avg[j][2]};
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    cov[r * 3 + c] += diff[r] * diff[c];
        }
        for (int i = 0; i < 9; i++) cov[i] /= (double)(sample_counts[j] - 1);
        invert3x3(cov, h_inv_cov[j]);
    }

    CSC(cudaMemcpyToSymbol(c_avg, h_avg, sizeof(double) * 32 * 3));
    CSC(cudaMemcpyToSymbol(c_inv_cov, h_inv_cov, sizeof(double) * 32 * 9));
    CSC(cudaMemcpyToSymbol(c_nc, &nc, sizeof(int)));

    uchar4* d_data;
    CSC(cudaMalloc(&d_data, sizeof(uchar4) * n));
    CSC(cudaMemcpy(d_data, h_data, sizeof(uchar4) * n, cudaMemcpyHostToDevice));

    classify_kernel<<<1024, 1024>>>(d_data, n);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(h_data, d_data, sizeof(uchar4) * n, cudaMemcpyDeviceToHost));

    fp = fopen(out_path, "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(h_data, sizeof(uchar4), n, fp);
    fclose(fp);

    CSC(cudaFree(d_data));
    for(int i = 0; i < nc; i++) free(samples[i]);
    free(samples); free(sample_counts); free(h_data);
    return 0;
}
