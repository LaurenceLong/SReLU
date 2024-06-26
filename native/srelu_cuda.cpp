#include <torch/extension.h>

torch::Tensor srelu_cuda_forward(torch::Tensor input, double t, double a, bool inplace);
torch::Tensor srelu_cuda_backward(torch::Tensor grad_output, torch::Tensor input, double t, double a);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &srelu_cuda_forward, "SReLU forward (CUDA)");
    m.def("backward", &srelu_cuda_backward, "SReLU backward (CUDA)");
}