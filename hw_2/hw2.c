#include <stdio.h>
#include <stdlib.h>

void swap(float *arr, int i, int j)
{
    float temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}

void bubbleSort(float arr[], int n)
{
    for (int i = 0; i < n - 1; i++)
    {

        // Last i elements are already in place, so the loop will only num n - i - 1 times
        for (int j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
                swap(arr, j, j + 1);
        }
    }
}


int main() {
  int n;

  if (scanf("%d", &n) != 1) {
    printf("ERROR: you need to write number: count of digits\n");
    return 0;
  }

  float* array = (float *)malloc(n * sizeof(float));
  if (array == NULL) {
    printf("ERROR: malloc :(((((");
    return 0;
  }
  for (int i = 0; i < n; i++) {
    if (scanf("%f", &array[i]) != 1) {
      printf("ERROR: you need to write float elem of array\n");
      return 0;
    }
  }

  bubbleSort(array, n);
  for (int i = 0; i < n; i++) {
    if ((n - i) == 1) {
      printf("%.6e\n", array[i]);
      break;
      
    }
    printf("%.6e ", array[i]);
  }

  free(array);
  return 0;
}
