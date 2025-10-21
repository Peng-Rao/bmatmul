// DON'T CHANGE THIS ^^ FILENAME!
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM 16

// utility for wrapping CUDA API calls and log any error they may return (use
// this for debugging)
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}

// === DO NOT CHANGE THIS ===
// host-side version, used to validate results
__host__ void batchedMatMulHost(float *M, float *N, float *P, int m, int k,
                                int n, int batch)
{
    for (int b = 0; b < batch; b++)
    {
        for (int row = 0; row < m; row++)
        {
            for (int col = 0; col < n; col++)
            {
                float value = 0.0f;
                for (int i = 0; i < k; i++)
                {
                    float a = M[row * k + i];
                    float c = N[b * (k * n) + i * n + col];
                    value += a * c;
                }
                P[b * (m * n) + row * n + col] = value;
            }
        }
    }
}

void initWith(float number, float *arr, int size)
{
    for (int i = 0; i < size; i++)
        arr[i] = number;
}

void initRandom(float *arr, int size, unsigned int seed, float minVal = 0.0f,
                float maxVal = 1.0f)
{
    srand(seed);
    for (int i = 0; i < size; i++)
    {
        float r = (float)rand() / RAND_MAX;
        arr[i] = minVal + r * (maxVal - minVal);
    }
}

void checkResult(float *arr1, float *arr2, int size)
{
    const float atol =
        1e-4f; // absolute tolerance for fp32 (lack of) associativity
    const float rtol =
        1e-4f; // relative tolerance for fp32 (lack of) associativity
    for (int i = 0; i < size; i++)
    {
        float diff = fabs(arr1[i] - arr2[i]);
        float tol = atol + rtol * fabs(arr2[i]);
        if (diff > tol)
        {
            printf("Error at %d: %f != %f (diff=%e, tol=%e)\n", i, arr1[i], arr2[i],
                   diff, tol);
            exit(1);
        }
    }
}
// ==========================

// this is the reference implementation
// you can change this to your heart's contempt
__global__ void batchedMatMul(float *M, float *N, float *P, int m, int k, int n,
                              int batch)
{
    int b = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Put tile into shared memory
    __shared__ float Ms[TILE_DIM][TILE_DIM];
    __shared__ float Ns[TILE_DIM][TILE_DIM];

    float value = 0.0f;

    for (int t = 0; t < (k + TILE_DIM - 1) / TILE_DIM; t++)
    {
        const int tiledRow = row;
        const int tiledCol = t * TILE_DIM + threadIdx.x;

        Ms[threadIdx.y][threadIdx.x] = (tiledRow < m && tiledCol < k) ? M[tiledRow * k + tiledCol] : 0.0f;

        // if (tiledRow < m && tiledCol < k)
        //     Ms[threadIdx.y][threadIdx.x] = M[tiledRow * k + tiledCol];
        // else
        //     Ms[threadIdx.y][threadIdx.x] = 0.0f;

        const int tiledRowN = t * TILE_DIM + threadIdx.y;
        const int tiledColN = col;

        Ns[threadIdx.y][threadIdx.x] = (tiledRowN < k && tiledColN < n) ? N[b * (k * n) + tiledRowN * n + tiledColN] : 0.0f;

        // if (tiledRowN < k && tiledColN < n)
        //     Ns[threadIdx.y][threadIdx.x] =
        //         N[b * (k * n) + tiledRowN * n + tiledColN];
        // else
        //     Ns[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

#pragma unroll
        for (int i = 0; i < TILE_DIM; i++)
            value += Ms[threadIdx.y][i] * Ns[i][threadIdx.x];

        __syncthreads();
    }

    if (row < m && col < n)
        P[b * (m * n) + row * n + col] = value;
}

int main(int argc, char **argv)
{
    // === DO NOT CHANGE THIS ===
    if (argc != 6)
    {
        printf("Usage: %s <m> <k> <n> <batch> <seed>\n", argv[0]);
        exit(1);
    }

    int m = atoi(argv[1]);     // rows of Ms and Ps
    int k = atoi(argv[2]);     // cols of Ms, rows of Ns
    int n = atoi(argv[3]);     // cols of Ns and Ps
    int batch = atoi(argv[4]); // number of matrix pairs
    unsigned int seed =
        (unsigned int)atoi(argv[5]); // seed for random initialization

    printf("Running batched matmul with m=%d, k=%d, n=%d, batch=%d, seed=%u\n", m,
           k, n, batch, seed);

    const int sizeM = m * k;
    const int sizeN = k * n * batch;
    const int sizeP = m * n * batch;

    float *M = (float *)malloc(sizeM * sizeof(float));
    float *N = (float *)malloc(sizeN * sizeof(float));
    float *P = (float *)malloc(sizeP * sizeof(float));

    initRandom(M, sizeM, seed);
    initRandom(N, sizeN, seed + 1);
    initWith(0.0f, P, sizeP);
    // ==========================

    // here, you can change anything
    float *M_d;
    float *N_d;
    float *P_d;

    gpuErrchk(cudaMalloc((void **)&M_d, sizeM * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&N_d, sizeN * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&P_d, sizeP * sizeof(float)));

    gpuErrchk(cudaMemcpy(M_d, M, sizeM * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(N_d, N, sizeN * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(P_d, P, sizeP * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockSize(TILE_DIM, TILE_DIM, 1);
    dim3 numBlocks((n + blockSize.x - 1) / blockSize.x,
                   (m + blockSize.y - 1) / blockSize.y, batch);

    batchedMatMul<<<numBlocks, blockSize>>>(M_d, N_d, P_d, m, k, n, batch);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(P, P_d, sizeP * sizeof(float), cudaMemcpyDeviceToHost));

    // === DO NOT CHANGE THIS ===
    // However: once you know results are correct, you can temporarily
    //          comment this out if you want to test performance on large
    //          matrices, since the evaluation on CPU can get pretty slow.
    printf("Checking results on CPU...\n");
    float *P_host = (float *)malloc(sizeP * sizeof(float));
    initWith(0.0f, P_host, sizeP);
    batchedMatMulHost(M, N, P_host, m, k, n, batch);
    checkResult(P, P_host, m * n * batch);
    printf("All results matched, success!");
    // ==========================

    // here, you can change anything, e.g. add some logging
    // logging occupancy
    cudaDeviceProp prop;
    gpuErrchk(cudaGetDeviceProperties(&prop, 0));

    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Warp size: %d\n", prop.warpSize);
    printf("  ------\n");
    printf("  Maximum GPU global memory size: %zu bytes\n", prop.totalGlobalMem);
    printf("  Maximum GPU constants memory size: %zu bytes\n",
           prop.totalConstMem);
    printf("  Maximum shared memory available per block: %zu bytes\n",
           prop.sharedMemPerBlock);
    printf("  Maximum shared memory available per SM: %zu bytes\n",
           prop.sharedMemPerMultiprocessor);
    printf("  ------\n");
    printf("  Maximum size of each dimension of a grid: %d x %d x %d\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Maximum size of each dimension of a block: %d x %d x %d\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Maximum number of threads per block: %d\n",
           prop.maxThreadsPerBlock);
    printf("  Number of 32-bit registers available per block: %d\n",
           prop.regsPerBlock);
    printf("  ------\n");
    printf("  Number of Streaming Multiprocessors: %d\n",
           prop.multiProcessorCount);
    printf("  Maximum number of resident blocks per SM: %d\n",
           prop.maxBlocksPerMultiProcessor);
    printf("  Maximum resident threads per SM: %d\n",
           prop.maxThreadsPerMultiProcessor);
    printf("  Number of 32-bit registers available per SM: %d\n",
           prop.regsPerMultiprocessor);

    int numBlocksPerSM = 0;
    gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM, batchedMatMul, blockSize.x * blockSize.y, 0));

    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
    int activeThreadsPerSM = numBlocksPerSM * blockSize.x * blockSize.y;

    // occupancy
    float occupancy = (float)activeThreadsPerSM / (float)maxThreadsPerSM * 100.0f;
    printf("\n=== Occupancy Report ===\n");
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Max threads per SM: %d\n", maxThreadsPerSM);
    printf("Block size: %d x %d = %d threads\n", blockSize.x, blockSize.y,
           blockSize.x * blockSize.y);
    printf("Active blocks per SM: %d\n", numBlocksPerSM);
    printf("Active threads per SM: %d\n", activeThreadsPerSM);
    printf("Occupancy: %.2f%%\n", occupancy);
    printf("=========================\n\n");

    gpuErrchk(cudaFree(M_d));
    gpuErrchk(cudaFree(N_d));
    gpuErrchk(cudaFree(P_d));

    free(M);
    free(N);
    free(P);
    free(P_host);

    return 0;
}
