#include <stdio.h>

__global__ void showWarpRanges()
{
    // Block configuration
    const int warpSize_ = 32;
    int blockWidth = blockDim.x;
    int blockHeight = blockDim.y;

    // Linear thread index within block
    int linear_id = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = linear_id / warpSize_;
    int thread_in_warp = linear_id % warpSize_;

    // 每个 warp 的第一个线程打印该 warp 起始 & 结束坐标
    if (thread_in_warp == 0)
    {
        int start_linear = warp_id * warpSize_;
        int end_linear = start_linear + warpSize_ - 1;

        // 转换为 (x, y)
        int start_x = start_linear % blockWidth;
        int start_y = start_linear / blockWidth;
        int end_x = end_linear % blockWidth;
        int end_y = end_linear / blockWidth;

        printf("Block (%d,%d): warp %d starts at (%2d,%2d), ends at (%2d,%2d)\n",
               blockIdx.x, blockIdx.y, warp_id, start_x, start_y, end_x, end_y);
    }
}

int main()
{
    dim3 block(24, 12); // 每个 block 有 288 个线程
    dim3 grid(16, 16);  // 任意网格维度

    showWarpRanges<<<grid, block>>>();
    cudaDeviceSynchronize();

    return 0;
}
