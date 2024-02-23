from torch.utils.cpp_extension import load_inline
import torch
import matplotlib.pyplot as plt

cuda_source = '''
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA kernel for the Mandelbrot set
__global__ void mandelbrot_set_kernel(float* output, int width, int height, float x_min, float x_max, float y_min, float y_max, int max_iter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < width && idy < height) {
        float x_scale = (x_max - x_min) / width;
        float y_scale = (y_max - y_min) / height;

        float c_re = x_min + idx * x_scale;
        float c_im = y_min + idy * y_scale;

        float zx = 0, zy = 0;
        int iter = 0;
        while (zx * zx + zy * zy < (2 * 2) && iter < max_iter) {
            float xtemp = zx * zx - zy * zy + c_re;
            zy = 2 * zx * zy + c_im;
            zx = xtemp;

            iter++;
        }

        // Normalize the iter value to be between 0 and 1
        output[idy * width + idx] = iter / (float)max_iter;
    }
}



// CUDA kernel wrapper for PyTorch
torch::Tensor mandelbrot_set_cuda(torch::Tensor output, int width, int height, float x_min, float x_max, float y_min, float y_max, int max_iter) {
    // Call the kernel here
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    dim3 threads(16, 16);
    mandelbrot_set_kernel<<<blocks, threads>>>(output.data_ptr<float>(), width, height, x_min, x_max, y_min, y_max, max_iter);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
'''

cpp_source = "torch::Tensor mandelbrot_set_cuda(torch::Tensor output, int width, int height, float x_min, float x_max, float y_min, float y_max, int max_iter);"

mandelbrot = load_inline(name='mandelbrot_set_module',
                    cuda_sources=cuda_source,
                    cpp_sources=cpp_source,
                    functions=['mandelbrot_set_cuda'],
                    verbose=True)

def mandelbrot_set(width, height, x_min, x_max, y_min, y_max, max_iter):
    """
    Generate a Mandelbrot set on the GPU using CUDA
        width: Width of the output image
        height: Height of the output image
        x_min: Minimum x value
        x_max: Maximum x value
        y_min: Minimum y value
        y_max: Maximum y value
        max_iter: Maximum number of iterations
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    output = torch.zeros(height, width, dtype=torch.float32, device=device)
    mandelbrot.mandelbrot_set_cuda(output, width, height, x_min, x_max, y_min, y_max, max_iter)
    return output.cpu().numpy()

def visualize_mandelbrot_set():
    # Example usage:
    width, height = 1024, 768
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    max_iter = 1000

    mandelbrot_fractal = mandelbrot_set(width, height, x_min, x_max, y_min, y_max, max_iter)

    plt.imshow(mandelbrot_fractal, cmap='hot')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('mandelbrot_set.png', dpi=1000)