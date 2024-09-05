#include<stdio.h>
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define THREAD_NUM 1024

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}


int getThreadNum()
{
    cudaDeviceProp prop;
    int count;

    HANDLE_ERROR(cudaGetDeviceCount(&count));
    printf("gpu num %d\n", count);
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
    printf("max thread num: %d\n", prop.maxThreadsPerBlock);
    printf("max grid dimensions: %d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    return prop.maxThreadsPerBlock;
}


// __global__ 函数 并行计算矩阵乘法
__global__ void matmult_kernel(const float* a, const float* b, float* c, int th, int n, int m, int d)
{
    //表示目前的 thread 是第几个 thread（由 0 开始计算）
    const int tid = threadIdx.x;

    //表示目前的 thread 属于第几个 block（由 0 开始计算）
    const int bid = blockIdx.x;

    //从 bid 和 tid 计算出这个 thread 应该计算的 row 和 column
    const int idx = bid * blockDim.x + tid;
    const int batch=idx/(n*d);
    const int row = (idx/d)%n;
    const int column =idx%d;

    int i;

    //计算矩阵乘法

    if (idx <(n*d*th))
    {
        float t = 0;
        for (i = 0; i < m; i++)
        {
            t += a[batch*n*m+row * m + i] - b[batch*m*d+i * m + column];
        }
	    c[batch * n * d + row * d + column] = exp(t/(-0.6 ));
    }
}


void launch_matmult(const float* a, const float* b, float* c, int th, int n, int m, int d) {
    //CUDA 初始化
    int blockNum = (th * n * d-0.5) / THREAD_NUM + 1;
    //int blockNum = (th * n * d + THREAD_NUM - 1) / THREAD_NUM;

    // 在CUDA 中执行函数 语法：函数名称<<<block 数目, thread 数目, shared memory 大小>>>(参数...);
    matmult_kernel <<<blockNum, THREAD_NUM>>> (a, b, c, th, n, m, d);
}