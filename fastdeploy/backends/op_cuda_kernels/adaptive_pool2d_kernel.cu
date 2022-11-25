#include "adaptive_pool2d_kernel.h"

namespace fastdeploy {

__global__ void CudaCastKernel(const float* in, float* out, int edge,  int out_bc_offset, int in_bc_offset, int ih, int iw, int oh, int ow, bool is_avg) {
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  if (position >= edge) {
    return;
  }
  int offset = floorf(float(position) / out_bc_offset);
  int h = floorf(float(position % out_bc_offset) / ow);
  int w = (position % out_bc_offset) % ow;
  int hstart = floorf(static_cast<float>(h * ih) / oh);
  int hend = ceilf(static_cast<float>((h + 1) * ih) / oh);
  int wstart = floorf(static_cast<float>(w * iw) / ow);
  int wend = ceilf(static_cast<float>((w + 1) * iw) / ow);
  if(is_avg) {
    out[position] = 0.0;
  } else {
    out[position] = in[offset * in_bc_offset + hstart * iw + wstart];
  }
  for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
      int input_idx = h * iw + w;
      if(is_avg) {
        out[position] = out[position] + in[offset * in_bc_offset + input_idx];
      } else {
        out[position] = max(out[position], in[offset * in_bc_offset + input_idx]);
      }
    }
  }
  out[position] = out[position] / ((hend - hstart) * (wend - wstart));
}

void CudaAdaptivePool(const std::vector<int64_t>& input_dims, const std::vector<int64_t>& output_dims, float* output, const float* input, void* compute_stream, const std::string& pooling_type){
  auto casted_compute_stream = reinterpret_cast<cudaStream_t>(compute_stream);
  int out_bc_offset = output_dims[2] * output_dims[3];
  int in_bc_offset = input_dims[2] * input_dims[3];
  int jobs = 1;
  for(int i : output_dims) {
    jobs *= i;
  }
  bool is_avg = pooling_type == "avg";
  int threads = 256;
  int blocks = ceil(jobs / static_cast<float>(threads));
  CudaCastKernel<<<blocks, threads, 0, casted_compute_stream>>>(
        input,
        output,
        jobs, out_bc_offset, in_bc_offset, int(input_dims[2]), int(input_dims[3]), int(output_dims[2]), int(output_dims[3]), is_avg);
}
}  // namespace fastdeploy