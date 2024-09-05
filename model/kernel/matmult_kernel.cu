#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include<stdio.h>
#include <float.h>
#include <math.h>
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define THREAD_NUM 1024

using namespace at;


// TODO make it in a common file
#define CUDA_KERNEL_LOOP_X(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)


namespace {
    const int CUDA_NUM_XTHREADS = 1024;
    const int CUDA_NUM_XYXTHREADS = 32;
    const int CUDA_NUM_XYYTHREADS = 32;
    const int kMaxGridNum = 65535;
    const int SDIM = 32;

    inline int GET_BLOCKS(const int N) {
        return std::min(kMaxGridNum, (N + CUDA_NUM_XTHREADS - 1) / CUDA_NUM_XTHREADS);
    }

    inline int GET_XBLOCKS(const int N) {
        return std::min(kMaxGridNum, (N + CUDA_NUM_XYXTHREADS - 1) / CUDA_NUM_XYXTHREADS);
    }

    inline int GET_YBLOCKS(const int N) {
        return std::min(kMaxGridNum, (N + CUDA_NUM_XYYTHREADS - 1) / CUDA_NUM_XYYTHREADS);
    }
}


// __global__ 函数 并行计算矩阵乘法
template <typename scalar_t>
__global__ void matmult_kernel(
        const scalar_t* query,
        const scalar_t* key,
        scalar_t* output,
        int xthreads,
        int batch_size,
        int num_head,
        int q_len,
        int k_len,
        int input_channels
        ) {
    CUDA_KERNEL_LOOP_X(index, xthreads) {
    //  printf("tid: %d \\n", index);
        const int b = index / num_head / q_len / k_len;
        const int h = (index / q_len / k_len) % num_head;
        const int p = (index / k_len) % q_len;
        const int q = index % k_len;
    //  printf("b: %d, h: %d, p: %d, q: %d \\n", b, h, p, q);
        if (index < batch_size * num_head * q_len * k_len){
            scalar_t sum = 0;
            for (int c = 0; c < input_channels; c++){
                int query_offset = b * (num_head * q_len * input_channels) + h * (q_len * input_channels) + p * input_channels + c;
                int key_offset = b * (num_head * k_len * input_channels) + h * (k_len * input_channels) + c * k_len + q;
                scalar_t dis = query[query_offset] - key[key_offset];
    //          printf("query off: %d, key off: %d, query data: %f, key data: %f, dis: %f \\n", query_offset, key_offset, query[query_offset], query[key_offset], dis);
//                 sum += dis * dis;
                sum += dis;
    //      printf("dis %f \\n", dis * dis);
            }
    //    printf("sum: %f \\n", sum);
            output[index] = sum;
        }
    }
}


void launch_matmult(
        const at::Tensor query,
        const at::Tensor key,
        at::Tensor output,
        int batch_size,
        int num_head,
        int q_len,
        int k_len,
        int input_channels) {
    const int nx = batch_size * num_head * q_len * k_len;
    at::cuda::CUDAGuard device_guard(query.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            query.scalar_type(), "subtraction_gaussian_forward_gpu", ([&] {
            const scalar_t* query_ = query.data_ptr<scalar_t>();
            const scalar_t* key_ = key.data_ptr<scalar_t>();
            scalar_t* output_ = output.data_ptr<scalar_t>();

            matmult_kernel<<<GET_BLOCKS(nx), CUDA_NUM_XTHREADS, 0, stream>>>(
                    query_,
                    key_,
                    output_,
                    nx,
                    batch_size,
                    num_head,
                    q_len,
                    k_len,
                    input_channels);
        }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf(
                "error in subtraction_gaussian_forward_cuda: %s\n",
                cudaGetErrorString(err));
    }
}
