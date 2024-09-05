#include <torch/types.h>


void launch_matmult(const at::Tensor query,
                    const at::Tensor key,
                    at::Tensor output,
                    int bs, int nh, int ql, int kl, int ic);
