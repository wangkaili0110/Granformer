#include <torch/extension.h>
#include "matmult.h"


void torch_launch_matmult(const at::Tensor query,
                       const at::Tensor key,
                       at::Tensor output,
                       int64_t bs, int64_t nh, int64_t ql, int64_t kl, int64_t ic) {

    launch_matmult(query, key, output, bs, nh, ql, kl, ic);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_matmult",
          &torch_launch_matmult,
          "matmult kernel warpper");
}

TORCH_LIBRARY(matmult, m) {
    m.def("torch_launch_matmult", torch_launch_matmult);
}
