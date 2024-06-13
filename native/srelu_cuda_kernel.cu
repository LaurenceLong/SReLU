#include <torch/extension.h>
#include <torch/types.h>
#include <cmath>
#include <cuda_fp16.h>  // 确保包含了正确的头文件

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
        half x2 = (idx + stride < num_elements) ? __ldg(&input[idx + stride]) : val_zero;

        half out1 = __hle(x1, t_neg) ? val_zero :
                    (__hge(x1, t) ? x1 :
                     __hdiv(__hmul(x1, __hadd(hsin(__hmul(a, x1)), val_one)), val_two));

        half out2 = __hle(x2, t_neg) ? val_zero :
                    (__hge(x2, t) ? x2 :
                     __hdiv(__hmul(x2, __hadd(hsin(__hmul(a, x2)), val_one)), val_two));

        output[idx] = out1;
        if (idx + stride < num_elements) {
            output[idx + stride] = out2;
        }

        idx += stride * 2;
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
        float x2 = (idx + stride < num_elements) ? __ldg(&input[idx + stride]) : val_zero;

        float out1 = (x1 <= -t) ? val_zero : ((x1 >= t) ? x1 : x1 * (sinf(a * x1) + val_one) / val_two);
        float out2 = (x2 <= -t) ? val_zero : ((x2 >= t) ? x2 : x2 * (sinf(a * x2) + val_one) / val_two);

        output[idx] = out1;
        if (idx + stride < num_elements) {
            output[idx + stride] = out2;
        }

        idx += stride * 2;
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
        double x2 = (idx + stride < num_elements) ? __ldg(&input[idx + stride]) : val_zero;

        double out1 = (x1 <= -t) ? val_zero : ((x1 >= t) ? x1 : x1 * (sin(a * x1) + val_one) / val_two);
        double out2 = (x2 <= -t) ? val_zero : ((x2 >= t) ? x2 : x2 * (sin(a * x2) + val_one) / val_two);

        output[idx] = out1;
        if (idx + stride < num_elements) {
            output[idx + stride] = out2;
        }

        idx += stride * 2;
    }
}

__global__ void srelu_backward_kernel_half(const half* __restrict__ input,
                            half* __restrict__ output,
                            int64_t num_elements,
                            const half t, const half a) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    const half val_zero = __float2half(0.0f);
    const half val_half = __float2half(0.5f);
    const half val_two = __float2half(2.0f);
    const half t_neg = __hneg(t);

    while (idx < num_elements) {
        half x1 = __ldg(&input[idx]);
        half x2 = (idx + stride < num_elements) ? __ldg(&input[idx + stride]) : val_zero;

        half a_mul_x1, sin_value1, cos_value1;
        half a_mul_x2, sin_value2, cos_value2;

        a_mul_x1 = __hmul(a, x1);
        sin_value1 = hsin(a_mul_x1);
        cos_value1 = hcos(a_mul_x1);

        a_mul_x2 = __hmul(a, x2);
        sin_value2 = hsin(a_mul_x2);
        cos_value2 = hcos(a_mul_x2);

        // 计算 out1 和 out2
        half out1 = __hle(x1, t_neg) ? val_zero :
                    (__hlt(x1, t) ?
                     __hmul(output[idx], __hadd(__hdiv(__hadd(__hmul(__hmul(a, x1), cos_value1), sin_value1), val_two), val_half)) :
                     output[idx]);

        half out2 = __hle(x2, t_neg) ? val_zero :
                    (__hlt(x2, t) ?
                     __hmul(output[idx + stride], __hadd(__hdiv(__hadd(__hmul(__hmul(a, x2), cos_value2), sin_value2), val_two), val_half)) :
                     output[idx + stride]);

        output[idx] = out1;
        if (idx + stride < num_elements) {
            output[idx + stride] = out2;
        }

        idx += stride * 2;
    }
}

__global__ void srelu_backward_kernel_float(const float* __restrict__ input,
                            float* __restrict__ output,
                            int64_t num_elements,
                            const float t, const float a) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    const float val_zero = 0.0f;
    const float val_half = 0.5f;
    const float val_two = 2.0f;

    while (idx < num_elements) {
        float x1 = __ldg(&input[idx]);
        float x2 = (idx + stride < num_elements) ? __ldg(&input[idx + stride]) : val_zero;

        float sin_value1, cos_value1;
        float sin_value2, cos_value2;

        sincosf(a * x1, &sin_value1, &cos_value1);
        sincosf(a * x2, &sin_value2, &cos_value2);

        float out1 = (x1 <= -t) ? val_zero : ((x1 < t) ? output[idx] * ((a * x1 * cos_value1 + sin_value1) / val_two + val_half) : output[idx]);
        float out2 = (x2 <= -t) ? val_zero : ((x2 < t) ? output[idx + stride] * ((a * x2 * cos_value2 + sin_value2) / val_two + val_half) : output[idx + stride]);

        output[idx] = out1;
        if (idx + stride < num_elements) {
            output[idx + stride] = out2;
        }

        idx += stride * 2;
    }
}

__global__ void srelu_backward_kernel_double(const double* __restrict__ input,
                            double* __restrict__ output,
                            int64_t num_elements,
                            const double t, const double a) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    const double val_zero = 0.0;
    const double val_half = 0.5;
    const double val_two = 2.0;

    while (idx < num_elements) {
        double x1 = __ldg(&input[idx]);
        double x2 = (idx + stride < num_elements) ? __ldg(&input[idx + stride]) : val_zero;

        double sin_value1, cos_value1;
        double sin_value2, cos_value2;

        sincos(a * x1, &sin_value1, &cos_value1);
        sincos(a * x2, &sin_value2, &cos_value2);

        double out1 = (x1 <= -t) ? val_zero : ((x1 < t) ? output[idx] * ((a * x1 * cos_value1 + sin_value1) / val_two + val_half) : output[idx]);
        double out2 = (x2 <= -t) ? val_zero : ((x2 < t) ? output[idx + stride] * ((a * x2 * cos_value2 + sin_value2) / val_two + val_half) : output[idx + stride]);

        output[idx] = out1;
        if (idx + stride < num_elements) {
            output[idx + stride] = out2;
        }

        idx += stride * 2;
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

        if (input.is_cuda()) {
            // Launch kernel
            scalar_t st = static_cast<scalar_t>(t);
            scalar_t sa = static_cast<scalar_t>(a);
            launch_srelu_forward_kernel(input_data, output_data, num_elements, st, sa);
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

        if (input.is_cuda()) {
            // Launch kernel
            scalar_t st = static_cast<scalar_t>(t);
            scalar_t sa = static_cast<scalar_t>(a);
            launch_srelu_backward_kernel(input_data, grad_input_data, num_elements, st, sa);
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