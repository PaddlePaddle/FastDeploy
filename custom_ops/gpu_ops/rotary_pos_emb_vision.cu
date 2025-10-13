#include <paddle/extension.h>
#include <vector>
#include <iostream>

// template <typename T>
constexpr int kBlockSize = 256;


// template <typename T>
__global__ void apply_rotary_pos_emb_kernel(
    const float* tensor,
    const float* freqs,
    const float* cos,
    const float* sin,
    float* output,
    int seq_len,
    int num_heads,
    int dim) {

    const int half_dim = dim / 2;
    const int total_elements = seq_len * num_heads * dim;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += blockDim.x * gridDim.x) {
        // printf("idx: %d\n", idx);
        // 计算三维索引
        const int seq_idx = idx / (num_heads * dim);
        const int head_idx = (idx / dim) % num_heads;
        const int dim_idx = idx % dim;

        // 执行旋转操作
        float x = tensor[idx];
        float cos_val = cos[seq_idx * dim + dim_idx];
        float sin_val = sin[seq_idx * dim + dim_idx];

        if (dim_idx < half_dim) {
            // 实际：output[..., :half] = x1 * cos + (-x2) * sin
            //       output[..., half:] = x2 * cos + x1 * sin
            // 所以对于 d < half，需要 x2 在 d + half 的值
            int x2_idx = seq_idx * num_heads * dim + head_idx * dim + (dim_idx + half_dim);
            float x2 = tensor[x2_idx];
            output[idx] = x * cos_val - x2 * sin_val;
        } else {
            int x1_idx = seq_idx * num_heads * dim + head_idx * dim + (dim_idx - half_dim);
            float x1 = tensor[x1_idx];
            output[idx] = x * cos_val + x1 * sin_val;
        }

    }
}

std::vector<paddle::Tensor> ApplyRotaryPosEmbVision(const paddle::Tensor& tensor,
                                                    const paddle::Tensor& freqs,
                                                    const paddle::Tensor& cos,
                                                    const paddle::Tensor& sin){
    auto cu_stream = tensor.stream();
    std::vector<int64_t> tensor_shape = tensor.shape();
    int max_seq_len = tensor.shape()[0];
    int num_heads = tensor.shape()[1];
    int dim = tensor.shape()[2];
    int block_size = kBlockSize;
    const int total_elements = max_seq_len * num_heads * dim;
    int grid_size = (total_elements + block_size - 1) / block_size;
    auto output = paddle::full(tensor.shape(), -1, tensor.dtype(), tensor.place());
    apply_rotary_pos_emb_kernel<<<grid_size, block_size, 0, cu_stream>>>(
        const_cast<float*>(tensor.data<float>()),
        const_cast<float*>(freqs.data<float>()),
        const_cast<float*>(cos.data<float>()),
        const_cast<float*>(sin.data<float>()),
        output.data<float>(),
        max_seq_len,
        num_heads,
        dim
    );

    return {output};
}

std::vector<std::vector<int64_t>> RotaryPosEmbInferShape(const std::vector<int64_t>& tensor_shape, const std::vector<int64_t>& freqs_shape) {
    std::vector<int64_t> out_shape = tensor_shape;
    return {out_shape};
}

std::vector<paddle::DataType> RotaryPosEmbInferDtype(const paddle::DataType& tensor_dtype, const paddle::DataType& freqs_dtype) {
    return {paddle::DataType::FLOAT32};
}

PD_BUILD_OP(apply_rotary_pos_emb_vision)
    .Inputs({"tensor", "freqs", "cos", "sin"})
    .Outputs({"output"})
    .SetKernelFn(PD_KERNEL(ApplyRotaryPosEmbVision))
    .SetInferShapeFn(PD_INFER_SHAPE(RotaryPosEmbInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RotaryPosEmbInferDtype));
