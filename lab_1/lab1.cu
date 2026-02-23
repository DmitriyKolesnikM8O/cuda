#include <stdio.h>

/*
Host = CPU
Device = GPU

GPU организует потоки в иерархию: Сетка (Grid) -> Блоки (Blocks) -> Потоки (Threads)
threadIdx.x (Индекс потока)
blockDim.x (Размер блока)
blockIdx.x (Индекс блока)
gridDim.x (Размер сетки)
*/
__global__ void elementsSquare(double *vec1_cuda, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  while(idx < n) {
        vec1_cuda[idx] = vec1_cuda[idx] * vec1_cuda[idx];
        idx += offset;
    }
}

int main() {

  //input

  int n;
  if (scanf("%d", &n) != 1) {
    fprintf(stderr, "ERROR: Where`s input?\n");
    return 0;
  }
  
  double *vec1 = (double*)malloc(n * sizeof(double));
  if (vec1 == NULL) {
    fprintf(stderr, "ERROR: Where`s memory for vec1?\n");
    return 0;
  }
  for (int i = 0; i < n; i++) {
    if (scanf("%lf", &vec1[i]) != 1) {
      fprintf(stderr, "ERROR: Where`s input for vec1?\n");
      return 0;
    }
  }

  //CUDA

  double *vec1_cuda;

  cudaMalloc(&vec1_cuda, n * sizeof(double));
  cudaMemcpy(vec1_cuda, vec1, n * sizeof(double), cudaMemcpyHostToDevice);

  elementsSquare<<<1024, 1024>>>(vec1_cuda, n);

  cudaMemcpy(vec1, vec1_cuda, sizeof(double) * n, cudaMemcpyDeviceToHost);
  
  //output

  for(int i = 0; i < n; i++) {
    if ((n - i) == 1) {
      printf("%.10e\n", vec1[i]);
      break;
    }
    printf("%.10e ", vec1[i]);
  }

  //free

  cudaFree(vec1_cuda);
  free(vec1);

  return 0;

}
