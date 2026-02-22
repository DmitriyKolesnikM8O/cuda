#include <stdio.h>
#include <math.h>

#define EPSILON 1e-6f


float clean_zero(float val) {
    if (fabsf(val) < EPSILON) return 0.0f;
    return val;
}

int main() {
    float a, b, c;

 
    if (scanf("%f %f %f", &a, &b, &c) != 3) {
        return 0; 
    }

 
    if (fabsf(a) < EPSILON) {
        if (fabsf(b) < EPSILON) {
            if (fabsf(c) < EPSILON) {
                printf("any\n"); // 0=0
            } else {
                printf("incorrect\n"); // 0=5
            }
        } else {
 
            float x = -c / b;
            printf("%.6f\n", clean_zero(x));
        }
        return 0;
    }

 
    float d = (b * b) - (4.0f * a * c);

    if (d < -EPSILON) {
 
        printf("imaginary\n");
    } else if (fabsf(d) <= EPSILON) {
 
        float x = -b / (2.0f * a);
        printf("%.6f\n", clean_zero(x));
    } else {
 
        float sqrtD = sqrtf(d);
        float x1 = (-b + sqrtD) / (2.0f * a);
        float x2 = (-b - sqrtD) / (2.0f * a);
        printf("%.6f %.6f\n", clean_zero(x1), clean_zero(x2));
    }

    return 0;
}
