
#include "indexer_topk.cuh"

#include <cuda_bf16.h>

#include "helper.h"
#include "append_attn/mem_util.cuh"
#include "append_attn/mma_tensor_op.cuh"
#include "append_attn/utils.cuh"
#include "paddle/extension.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/utils/optional.h"


// using namespace flashinfer;
#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

template<paddle::DataType T>
cudaError_t DispatchTopK(
    paddle::Tensor& input,
    paddle::Tensor& output_indices,
    const paddle::Tensor& offsets,
    paddle::Tensor& lengths,
    uint32_t num_rows,
    paddle::Tensor& seq_len,
    paddle::Tensor& batch_id_per_token,
    uint32_t top_k,
    uint32_t max_len,
    flashinfer::sampling::RadixRowState* row_states_ptr,
    cudaStream_t stream) {

   typedef PDTraits<T> traits_;
   typedef typename traits_::DataType DataType_;
   typedef typename traits_::data_t data_t;

   cudaError_t status;
   status = flashinfer::sampling::TopKRaggedTransformDispatch<DataType_,int32_t>(
         reinterpret_cast<DataType_*>(input.data<data_t>()),
         static_cast<int32_t*>(output_indices.data<int32_t>()),
         static_cast<const int32_t*>(offsets.data<int32_t>()),
         static_cast<int32_t*>(lengths.data<int32_t>()),
         num_rows, 
         static_cast<int32_t*>(seq_len.data<int32_t>()),
         static_cast<int32_t*>(batch_id_per_token.data<int32_t>()),
         static_cast<uint32_t>(top_k), 
         max_len, 
         row_states_ptr, 
         stream);
   return status;
}

void RadixTopkRaggedTransform(
    paddle::Tensor& input,
    paddle::Tensor& output_indices,
    const paddle::Tensor& offsets,
    paddle::Tensor& lengths,
    paddle::Tensor& seq_len,
    paddle::Tensor& batch_id_per_token,
    paddle::optional<paddle::Tensor>& maybe_row_states_buffer,
    int top_k) {

//   CHECK_INPUT(input);
//   CHECK_INPUT(output_indices);
//   CHECK_INPUT(offsets);
//   CHECK_INPUT(lengths);
//   CHECK_DIM(2, input);           // input: (num_rows, max_len)
//   CHECK_DIM(2, output_indices);  // output_indices: (num_rows, top_k)
//   CHECK_DIM(1, offsets);         // offsets: (num_rows,)
//   CHECK_DIM(1, lengths);         // lengths: (num_rows,)

  unsigned int num_rows = input.dims()[0];
  unsigned int max_len = input.dims()[1];

  static cudaStream_t  stream = input.stream();
  cudaError_t status;
  auto input_dtype = input.dtype();

//   sampling::RadixRowState* row_states_ptr = nullptr;
//   if (maybe_row_states_buffer.has_value()) {
//     row_states_ptr =
//         static_cast<sampling::RadixRowState*>(maybe_row_states_buffer.value().data_ptr());
//   }
   flashinfer::sampling::RadixRowState* row_states_ptr = nullptr;
   if(maybe_row_states_buffer){
      auto& tensor_ptr = maybe_row_states_buffer.get();
      row_states_ptr = reinterpret_cast<flashinfer::sampling::RadixRowState*>(tensor_ptr.data<uint8_t>());
   }

   if (input_dtype == paddle::DataType::BFLOAT16) {
      status = DispatchTopK<paddle::DataType::BFLOAT16>(
         input, output_indices, offsets, lengths,
         num_rows,seq_len, batch_id_per_token, top_k, max_len, row_states_ptr, stream);
   } else if (input_dtype == paddle::DataType::FLOAT32) {
      status = DispatchTopK<paddle::DataType::FLOAT32>(
         input, output_indices, offsets, lengths,
         num_rows, seq_len, batch_id_per_token, top_k, max_len, row_states_ptr, stream);
   }
}


PD_BUILD_STATIC_OP(radix_topk_ragged_transform)
    .Inputs({"input", 
             "output_indices",
             "offsets",
             "lengths",
             "seq_len",
             "batch_id_per_token",
             paddle::Optional("maybe_row_states_buffer")})
    .Attrs({"top_k : int"})
    .SetKernelFn(PD_KERNEL(RadixTopkRaggedTransform));
