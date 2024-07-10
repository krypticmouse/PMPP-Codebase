#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__
void matmul(int* A, int* B, int* C, int M, int N, int O) { 
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int total = 0;

    if (row<M && col<O){
        for (int i=0; i<N; i++)
            total += A[row*N + i] * B[i*O + col];

        C[row * O + col] = total;
    }
}

int main() {
    int M = 4, N = 3, O = 3;

    int A_h[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    int B_h[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    int *C_h = (int*)malloc(M * O * sizeof(int));
    int *A_d, *B_d, *C_d;

    cudaMalloc((void**) &A_d, sizeof(int) * M * N);
    cudaMalloc((void**) &B_d, sizeof(int) * N * O);
    cudaMalloc((void**) &C_d, sizeof(int) * M * O);

    cudaMemcpy(
        A_d, A_h,
        sizeof(int) * M * N,
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        B_d, B_h,
        sizeof(int) * N * O,
        cudaMemcpyHostToDevice
    );

    dim3 block_dims = dim3(1, 1);
    dim3 thread_dims = dim3(M, O);

    matmul<<<block_dims, thread_dims>>>(A_d, B_d, C_d, M, N, O);
    cudaDeviceSynchronize();

    free(A_h);
    free(B_h);
    cudaFree(A_d);
    cudaFree(B_d);
    
    cudaMemcpy(
        C_h, C_d,
        sizeof(int) * M * O,
        cudaMemcpyDeviceToHost
    );

    for (int i=0; i<M; i++) {
        for (int j=0; j<O; j++) {
            cout<<C_h[i * O + j]<<" ";
        }
        cout<<endl;
    }

    free(C_h);
    cudaFree(C_d);

    return 0;
}