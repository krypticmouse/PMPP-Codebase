#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__
void add_vectors(float *a, float *b, float *c, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int N = 10;
    
    float *a_h, *b_h, *c_h;    // Host Memory Vectors
    float *a_d, *b_d, *c_d;    // Device Memory Vectors

    a_h = (float*)malloc(sizeof(float) * N);
    b_h = (float*)malloc(sizeof(float) * N);
    c_h = (float*)malloc(sizeof(float) * N);

    for (int i=0; i<N; i++){
        a_h[i] = i;
        b_h[i] = 2*i;
    }

    cudaMalloc((void**) &a_d, sizeof(float) * N);
    cudaMalloc((void**) &b_d, sizeof(float) * N);
    cudaMalloc((void**) &c_d, sizeof(float) * N);

    cudaMemcpy(
        a_d, a_h,
        sizeof(float) * N,
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        b_d, b_h,
        sizeof(float) * N,
        cudaMemcpyHostToDevice
    );

    add_vectors<<<1, N>>>(a_d, b_d, c_d, N);

    cudaFree(a_d);
    cudaFree(b_d);
    free(a_h);
    free(b_h);

    cudaMemcpy(
        c_h, c_d,
        sizeof(float) * N,
        cudaMemcpyDeviceToHost
    );

    cout<<"[";
    for (int i=0;i<N; i++) {
        cout<<c_h[i]<<",";
    }
    cout<<"]"<<endl;

    cudaFree(c_d);
    free(c_h);
}
