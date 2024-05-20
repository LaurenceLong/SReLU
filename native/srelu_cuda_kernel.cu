#include <torch/extension.h>
#include <torch/types.h>
#include <cmath>

template <typename scalar_t>
__global__ void srelu_kernel_forward(const scalar_t* __restrict__ input,
                            scalar_t* __restrict__ output,
                            int64_t num_elements,
                            float t, float a) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    scalar_t val_zero = static_cast<scalar_t>(0);
    scalar_t val_one = static_cast<scalar_t>(1);
    scalar_t val_two = static_cast<scalar_t>(2);

    while (idx < num_elements) {
        scalar_t x = input[idx];

        if (x <= -t) {
            output[idx] = val_zero;
        } else if (x >= t) {
            output[idx] = x;
        } else {
            if constexpr (std::is_same<scalar_t, half>::value) {
                output[idx] = x * (hsin(a * x) + val_one) / val_two;
            } else {
                output[idx] = x * (sinf(a * x) + val_one) / val_two;
            }
        }

        idx += stride;
    }
}

template <typename scalar_t>
void launch_srelu_kernel_forward(const scalar_t* input,
                                scalar_t* output,
                                int64_t num_elements,
                                float t, float a) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    srelu_kernel_forward<<<blocks, threads>>>(input, output, num_elements, t, a);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(error)));
    }

    cudaDeviceSynchronize();
}

torch::Tensor srelu_cuda_forward(torch::Tensor input, float t, float a, bool inplace) {
    // Ensure input is a floating point tensor
    if (!input.is_floating_point()) {
        throw std::runtime_error("Input tensor must be a floating point tensor");
    }

    // Create an output tensor with the same properties as input
    auto output = input;

    if (!inplace) {
        output = torch::empty_like(input);
    }

    int64_t num_elements = input.numel();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "srelu_cuda_forward", ([&] {
        const scalar_t* input_data = input.data_ptr<scalar_t>();
        scalar_t* output_data = output.data_ptr<scalar_t>();

        if (input.is_cuda()) {
            // Launch kernel
            launch_srelu_kernel_forward(input_data, output_data, num_elements, t, a);
        } else {
            // Compute values
            for (int64_t idx = 0; idx < num_elements; idx++) {
                scalar_t x = input_data[idx];
                if (x <= -t) {
                    output_data[idx] = 0;
                } else if (x >= t) {
                    output_data[idx] = x;
                } else{
                    output_data[idx] = x * (sin(a * x) + 1) / 2;
                }
            }
        }
    }));

    return output;
}


template <typename scalar_t>
__global__ void srelu_kernel_backward(const scalar_t* __restrict__ input,
                            scalar_t* __restrict__ output,
                            int64_t num_elements,
                            float t, float a) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    scalar_t val_zero = static_cast<scalar_t>(0);
    scalar_t val_half = static_cast<scalar_t>(0.5);
    scalar_t val_two = static_cast<scalar_t>(2);

    while (idx < num_elements) {
        scalar_t x = input[idx];

        if (x <= -t) {
            output[idx] = val_zero;
        } else if (x < t) {
            if constexpr (std::is_same<scalar_t, half>::value) {
                output[idx] *= (a * x * hcos(a * x) + hsin(a * x)) / val_two + val_half;
            } else {
                output[idx] *= (a * x * cosf(a * x)  + sinf(a * x)) / val_two + val_half;
            }
        }

        idx += stride;
    }
}

template <typename scalar_t>
void launch_srelu_kernel_backward(const scalar_t* input,
                                scalar_t* output,
                                int64_t num_elements,
                                float t, float a) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    srelu_kernel_backward<<<blocks, threads>>>(input, output, num_elements, t, a);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(error)));
    }

    cudaDeviceSynchronize();
}

torch::Tensor srelu_cuda_backward(torch::Tensor grad_output, torch::Tensor input, float t, float a) {
    // Ensure input is a floating point tensor
    if (!grad_output.is_floating_point() || !input.is_floating_point()) {
        throw std::runtime_error("Input tensor must be a floating point tensor");
    }

    // Create an grad_input tensor with the same properties as grad_output
    auto grad_input = grad_output.clone();
    int64_t num_elements = grad_output.numel();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "srelu_cuda_backward", ([&] {
        const scalar_t* input_data = input.data_ptr<scalar_t>();
        scalar_t* grad_input_data = grad_input.data_ptr<scalar_t>();

        if (input.is_cuda()) {
            // Launch kernel
            launch_srelu_kernel_backward(input_data, grad_input_data, num_elements, t, a);
        } else {
            // Compute values
            for (int64_t idx = 0; idx < num_elements; idx++) {
                scalar_t x = input_data[idx];
                if (x <= -t) {
                    grad_input_data[idx] = 0;
                } else if (x < t) {
                    grad_input_data[idx] *= a * x * cos(a * x) / 2 + sin(a * x) / 2 + 1 / 2;
                }
            }
        }
    }));

    return grad_input;
}