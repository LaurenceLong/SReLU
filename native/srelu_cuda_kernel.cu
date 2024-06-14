#include <torch/extension.h>
#include <torch/types.h>
#include <cmath>
#include <cuda_fp16.h>
#include <omp.h>


__global__ void srelu_forward_kernel_half(const half* __restrict__ input,
                            half* __restrict__ output,
                            int64_t num_elements,
                            const half t, const half a) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    const half val_zero = __float2half(0.0f);
    const half val_one = __float2half(1.0f);
    const half val_two = __float2half(2.0f);
    const half t_neg = __hneg(t);

    while (idx < num_elements) {
        half x1 = __ldg(&input[idx]);

        if (__hle(x1, t_neg)) {
            output[idx] = val_zero;
        } else if (__hlt(x1, t)) {
            output[idx] = __hdiv(__hmul(x1, __hadd(hsin(__hmul(a, x1)), val_one)), val_two);
        } else {
            output[idx] = x1;
        }

        idx += stride;
    }
}

__global__ void srelu_forward_kernel_float(const float* __restrict__ input,
                            float* __restrict__ output,
                            int64_t num_elements,
                            const float t, const float a) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    const float val_zero = 0.0f;
    const float val_one = 1.0f;
    const float val_two = 2.0f;

    while (idx < num_elements) {
        float x1 = __ldg(&input[idx]);

        if (x1 <= -t) {
            output[idx] = val_zero;
        } else if (x1 < t) {
            output[idx] = x1 * (sinf(a * x1) + val_one) / val_two;
        } else {
            output[idx] = x1;
        }

        idx += stride;
    }
}

__global__ void srelu_forward_kernel_double(const double* __restrict__ input,
                            double* __restrict__ output,
                            int64_t num_elements,
                            const double t, const double a) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    const double val_zero = 0.0;
    const double val_one = 1.0;
    const double val_two = 2.0;

    while (idx < num_elements) {
        double x1 = __ldg(&input[idx]);

        if (x1 <= -t) {
            output[idx] = val_zero;
        } else if (x1 < t) {
            output[idx] = x1 * (sin(a * x1) + val_one) / val_two;
        } else {
            output[idx] = x1;
        }

        idx += stride;
    }
}

__global__ void srelu_backward_kernel_half(const half* __restrict__ input,
                            half* __restrict__ output,
                            int64_t num_elements,
                            const half t, const half a) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    const half val_zero = __float2half(0.0f);
    const half val_one = __float2half(1.0f);
    const half val_two = __float2half(2.0f);
    const half t_neg = __hneg(t);

    while (idx < num_elements) {
        half x1 = __ldg(&input[idx]);

        const half a_mul_x1 = __hmul(a, x1);
        half sin_value1, cos_value1;

        sin_value1 = hsin(a_mul_x1);
        cos_value1 = hcos(a_mul_x1);

        if (__hle(x1, t_neg)) {
            output[idx] = val_zero;
        } else if (__hlt(x1, t)) {
            output[idx] = __hmul(output[idx], __hdiv(__hadd(__hadd(__hmul(a_mul_x1, cos_value1), sin_value1), val_one), val_two));
        }

        idx += stride;
    }
}

__global__ void srelu_backward_kernel_float(const float* __restrict__ input,
                            float* __restrict__ output,
                            int64_t num_elements,
                            const float t, const float a) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    const float val_zero = 0.0f;
    const float val_one = 1.0f;
    const float val_two = 2.0f;

    while (idx < num_elements) {
        float x1 = __ldg(&input[idx]);

        const float a_mul_x1 = a * x1;
        float sin_value1, cos_value1;

        sincosf(a_mul_x1, &sin_value1, &cos_value1);

        if (x1 <= -t) {
            output[idx] = val_zero;
        } else if (x1 < t) {
            output[idx] *= (a_mul_x1 * cos_value1 + sin_value1 + val_one) / val_two;
        }

        idx += stride;
    }
}

__global__ void srelu_backward_kernel_double(const double* __restrict__ input,
                            double* __restrict__ output,
                            int64_t num_elements,
                            const double t, const double a) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    const double val_zero = 0.0;
    const double val_one = 1.0;
    const double val_two = 2.0;

    while (idx < num_elements) {
        double x1 = __ldg(&input[idx]);

        const double a_mul_x1 = a * x1;
        double sin_value1, cos_value1;

        sincos(a_mul_x1, &sin_value1, &cos_value1);

        if (x1 <= -t) {
            output[idx] = val_zero;
        } else if (x1 < t) {
            output[idx] *= (a_mul_x1 * cos_value1 + sin_value1 + val_one) / val_two;
        }

        idx += stride;
    }
}

template <typename scalar_t>
void launch_srelu_forward_kernel(const scalar_t* input,
                                scalar_t* output,
                                int64_t num_elements,
                                scalar_t t, scalar_t a) {
    int threads = 256; // Start with a common optimal value
    int blocks = (num_elements + threads - 1) / threads;

    if constexpr (std::is_same<scalar_t, at::Half>::value) {
        srelu_forward_kernel_half<<<blocks, threads>>>(
            reinterpret_cast<const half*>(input),
            reinterpret_cast<half*>(output),
            num_elements, static_cast<half>(t), static_cast<half>(a));
    } else if constexpr (std::is_same<scalar_t, float>::value) {
        srelu_forward_kernel_float<<<blocks, threads>>>(
            reinterpret_cast<const float*>(input),
            reinterpret_cast<float*>(output),
            num_elements, static_cast<float>(t), static_cast<float>(a));
    } else if constexpr (std::is_same<scalar_t, double>::value) {
        srelu_forward_kernel_double<<<blocks, threads>>>(
            reinterpret_cast<const double*>(input),
            reinterpret_cast<double*>(output),
            num_elements, static_cast<double>(t), static_cast<double>(a));
    } else {
        throw std::runtime_error("Unsupported data type");
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(error)));
    }

    cudaDeviceSynchronize();
}

template <typename scalar_t>
void launch_srelu_backward_kernel(const scalar_t* input,
                                scalar_t* output,
                                int64_t num_elements,
                                scalar_t t, scalar_t a) {
    int threads = 256; // Start with a common optimal value
    int blocks = (num_elements + threads - 1) / threads;

    if constexpr (std::is_same<scalar_t, at::Half>::value) {
        srelu_backward_kernel_half<<<blocks, threads>>>(
            reinterpret_cast<const half*>(input),
            reinterpret_cast<half*>(output),
            num_elements, static_cast<half>(t), static_cast<half>(a));
    } else if constexpr (std::is_same<scalar_t, float>::value) {
        srelu_backward_kernel_float<<<blocks, threads>>>(
            reinterpret_cast<const float*>(input),
            reinterpret_cast<float*>(output),
            num_elements, static_cast<float>(t), static_cast<float>(a));
    } else if constexpr (std::is_same<scalar_t, double>::value) {
        srelu_backward_kernel_double<<<blocks, threads>>>(
            reinterpret_cast<const double*>(input),
            reinterpret_cast<double*>(output),
            num_elements, static_cast<double>(t), static_cast<double>(a));
    } else {
        throw std::runtime_error("Unsupported data type");
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(error)));
    }

    cudaDeviceSynchronize();
}


torch::Tensor srelu_cuda_forward(torch::Tensor input, double t, double a, bool inplace) {
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

        scalar_t st = static_cast<scalar_t>(t);
        scalar_t sa = static_cast<scalar_t>(a);
        if (input.is_cuda()) {
            // Launch kernel
            launch_srelu_forward_kernel(input_data, output_data, num_elements, st, sa);
        } else {
            // Compute values in parallel
            #pragma omp parallel for
            for (int64_t idx = 0; idx < num_elements; idx++) {
                scalar_t x = input_data[idx];
                if (x <= -st) {
                    output_data[idx] = 0;
                } else if (x < st) {
                    output_data[idx] = x * (sin(sa * x) + 1) / 2;
                } else {
                    output_data[idx] = x;
                }
            }
        }
    }));

    return output;
}

torch::Tensor srelu_cuda_backward(torch::Tensor grad_output, torch::Tensor input, double t, double a) {
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

        scalar_t st = static_cast<scalar_t>(t);
        scalar_t sa = static_cast<scalar_t>(a);
        if (input.is_cuda()) {
            // Launch kernel
            launch_srelu_backward_kernel(input_data, grad_input_data, num_elements, st, sa);
        } else {
            if (std::is_same<scalar_t, float>::value) {
                // Compute values in parallel for float type
                #pragma omp parallel for
                for (int64_t idx = 0; idx < num_elements; idx++) {
                    scalar_t x = input_data[idx];
                    if (x <= -st) {
                        grad_input_data[idx] = 0;
                    } else if (x < st) {
                        float ax = static_cast<float>(sa * x);
                        float sin_ax, cos_ax;
                        sincos(ax, &sin_ax, &cos_ax);
                        grad_input_data[idx] *= (ax * cos_ax + sin_ax + 1) / 2;
                    }
                }
            } else if (std::is_same<scalar_t, double>::value) {
                // Compute values in parallel for double type
                #pragma omp parallel for
                for (int64_t idx = 0; idx < num_elements; idx++) {
                    scalar_t x = input_data[idx];
                    if (x <= -st) {
                        grad_input_data[idx] = 0;
                    } else if (x < st) {
                        double ax = static_cast<double>(sa * x);
                        double sin_ax, cos_ax;
                        sincos(ax, &sin_ax, &cos_ax);
                        grad_input_data[idx] *= (ax * cos_ax + sin_ax + 1) / 2;
                    }
                }
            } else {
                // Compute values in parallel for other types
                #pragma omp parallel for
                for (int64_t idx = 0; idx < num_elements; idx++) {
                    scalar_t x = input_data[idx];
                    if (x <= -st) {
                        grad_input_data[idx] = 0;
                    } else if (x < st) {
                        scalar_t ax = sa * x;
                        scalar_t sin_ax = std::sin(ax);
                        scalar_t cos_ax = std::cos(ax);
                        grad_input_data[idx] *= (ax * cos_ax + sin_ax + 1) / 2;
                    }
                }
            }
        }
    }));

    return grad_input;
}