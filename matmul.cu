#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// =========================
// 朴素矩阵乘法 Kernel
// =========================
__global__ void matmul_naive(float *A, float *B, float *C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// =========================
// Tile 优化矩阵乘法 Kernel
// =========================
__global__ void matmul_tiled(float *A, float *B, float *C, int M, int N, int K)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t)
    {
        int A_col = t * TILE_SIZE + threadIdx.x;
        int B_row = t * TILE_SIZE + threadIdx.y;

        // 载入共享内存
        As[threadIdx.y][threadIdx.x] = (row < M && A_col < K) ? A[row * K + A_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (col < N && B_row < K) ? B[B_row * N + col] : 0.0f;

        __syncthreads();

        // 累加当前 tile
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// =========================
// Host 辅助函数
// =========================
void random_init(float *mat, int size)
{
    for (int i = 0; i < size; ++i)
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Usage: %s [naive|tiled]\n", argv[0]);
        return 1;
    }

    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);

    random_init(h_A, M * K);
    random_init(h_B, K * N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // =========================
    // 运行 Kernel
    // =========================
    cudaEventRecord(start);

    if (strcmp(argv[1], "naive") == 0)
    {
        matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    else if (strcmp(argv[1], "tiled") == 0)
    {
        matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    else
    {
        printf("Unknown mode: %s\n", argv[1]);
        return 1;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Kernel mode: %s\n", argv[1]);
    printf("Execution time: %.3f ms\n", ms);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // =========================
    // 清理资源
    // =========================
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
