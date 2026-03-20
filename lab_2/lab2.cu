#include <stdio.h>
#include <stdlib.h>

#define CSC(call) { \
    cudaError_t res = call; \
    if (res != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(res)); \
        exit(0); \
    } \
}

__device__ float get_gray(uchar4 p) {
    return 0.299f * p.x + 0.587f * p.y + 0.114f * p.z;
}

__global__ void roberts_kernel(cudaTextureObject_t tex, uchar4* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    float i00 = get_gray(tex2D<uchar4>(tex, x, y));
    float i11 = get_gray(tex2D<uchar4>(tex, x + 1, y + 1));
    float i10 = get_gray(tex2D<uchar4>(tex, x + 1, y));
    float i01 = get_gray(tex2D<uchar4>(tex, x, y + 1));

    float g = sqrtf(powf(i00 - i11, 2.0f) + powf(i10 - i01, 2.0f));

    unsigned char res = (unsigned char)fminf(255.0f, g);

    uchar4 p_orig = tex2D<uchar4>(tex, x, y);
    out[y * w + x] = make_uchar4(res, res, res, p_orig.w);
}

int main() {
    char input_path[256], output_path[256];
    if (scanf("%s %s", input_path, output_path) != 2) return 0;

    FILE* fp = fopen(input_path, "rb");
    if (!fp) return 0;

    int w, h;
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);

    size_t size = (size_t)w * h;
    uchar4* h_data = (uchar4*)malloc(sizeof(uchar4) * size);
    fread(h_data, sizeof(uchar4), size, fp);
    fclose(fp);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray_t cuArray;
    CSC(cudaMallocArray(&cuArray, &channelDesc, w, h));
    CSC(cudaMemcpy2DToArray(cuArray, 0, 0, h_data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp; 
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    uchar4* d_out;
    CSC(cudaMalloc(&d_out, sizeof(uchar4) * size));

    dim3 threads(16, 16);
    dim3 blocks((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);
    
    roberts_kernel<<<blocks, threads>>>(tex, d_out, w, h);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(h_data, d_out, sizeof(uchar4) * size, cudaMemcpyDeviceToHost));

    fp = fopen(output_path, "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(h_data, sizeof(uchar4), size, fp);
    fclose(fp);

    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(cuArray));
    CSC(cudaFree(d_out));
    free(h_data);

    return 0;
}
