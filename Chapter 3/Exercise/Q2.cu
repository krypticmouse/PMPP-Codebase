#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__
void vecmatmul(int *a, int *B, int *c, int M) {
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int total = 0;

    if (col < M) {
        for (int i = 0; i < M; i++)
            total += a[i] * B[i * M + col];

        c[col] = total;
    }
}

int main() {
    int M = 3;
    int a_h[] = {1, 2, 3};
    int B_h[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    int *c_h, *a_d, *B_d, *c_d;

    c_h = (int*)malloc(sizeof(int) * M);

    cudaMalloc((void**) &a_d, sizeof(int) * M);
    cudaMalloc((void**) &B_d, sizeof(int) * M * M);
    cudaMalloc((void**) &c_d, sizeof(int) * M);
    
    cudaMemcpy(
        a_d, a_h,
        sizeof(int) * M,
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        B_d, B_h,
        sizeof(int) * M * M,
        cudaMemcpyHostToDevice
    );

    vecmatmul<<<1, M>>>(a_d, B_d, c_d, M);
    cudaDeviceSynchronize();

    cudaMemcpy(
        c_h, c_d,
        sizeof(int) * M,
        cudaMemcpyDeviceToHost
    );

    cudaFree(a_d);
    cudaFree(B_d);

    for (int i=0; i<M; i++)
        cout<<c_h[i]<<" ";
    cout<<endl;

    free(c_h);
    cudaFree(c_d);

    return 0;
}