
#include "indexer_topk.cuh"

#include <cuda_bf16.h>

#include "paddle/extension.h"

#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/utils/optional.h"

#include "append_attn/mem_util.cuh"
#include "append_attn/mma_tensor_op.cuh"
#include "append_attn/utils.cuh"
#include "helper.h"

// using namespace flashinfer;
#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

template <paddle::DataType T>
cudaError_t DispatchTopK(paddle::Tensor& input,
                         paddle::Tensor& output_indices,
                         const paddle::Tensor& offsets,
                         paddle::Tensor& lengths,
                         uint32_t num_rows,
                         const int32_t* seq_len_decoder,
                         const int32_t* batch_id_per_token,
                         const int32_t* block_tables,
                         uint32_t max_block_num,
                         uint32_t top_k,
                         uint32_t q_num_heads,
                         uint32_t max_len,
                         flashinfer::sampling::RadixRowState* row_states_ptr,
                         cudaStream_t stream) {
  typedef PDTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  cudaError_t status;
  status =
      flashinfer::sampling::TopKRaggedTransformDispatch<DataType_, int32_t>(
          reinterpret_cast<DataType_*>(input.data<data_t>()),
          static_cast<int32_t*>(output_indices.data<int32_t>()),
          static_cast<const int32_t*>(offsets.data<int32_t>()),
          static_cast<int32_t*>(lengths.data<int32_t>()),
          num_rows,
          seq_len_decoder,
          batch_id_per_token,
          block_tables,
          static_cast<uint32_t>(max_block_num),
          static_cast<uint32_t>(top_k),
          static_cast<uint32_t>(q_num_heads),
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
    paddle::optional<paddle::Tensor>& seq_len_decoder,
    paddle::optional<paddle::Tensor>& batch_id_per_token,
    paddle::optional<paddle::Tensor>& block_tables,
    paddle::optional<paddle::Tensor>& maybe_row_states_buffer,
    int max_block_num,
    int top_k,
    int q_num_heads = 0) {
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

  cudaStream_t stream = input.stream();
  cudaError_t status;
  auto input_dtype = input.dtype();

  //   sampling::RadixRowState* row_states_ptr = nullptr;
  //   if (maybe_row_states_buffer.has_value()) {
  //     row_states_ptr =
  //         static_cast<sampling::RadixRowState*>(maybe_row_states_buffer.value().data_ptr());
  //   }
  flashinfer::sampling::RadixRowState* row_states_ptr = nullptr;
  if (maybe_row_states_buffer) {
    auto& tensor_ptr = maybe_row_states_buffer.get();
    row_states_ptr = reinterpret_cast<flashinfer::sampling::RadixRowState*>(
        tensor_ptr.data<uint8_t>());
  }

  const int32_t* seq_len_ptr = nullptr;
  if (seq_len_decoder) {
    auto& tensor_ptr = seq_len_decoder.get();
    seq_len_ptr = static_cast<const int32_t*>(tensor_ptr.data<int32_t>());
  }
  const int32_t* batch_id_per_token_ptr = nullptr;
  if (batch_id_per_token) {
    auto& tensor_ptr = batch_id_per_token.get();
    batch_id_per_token_ptr =
        static_cast<const int32_t*>(tensor_ptr.data<int32_t>());
  }
  const int32_t* block_tables_ptr = nullptr;
  if (block_tables) {
    auto& tensor_ptr = block_tables.get();
    block_tables_ptr = static_cast<const int32_t*>(tensor_ptr.data<int32_t>());
  }

  if (input_dtype == paddle::DataType::BFLOAT16) {
    status = DispatchTopK<paddle::DataType::BFLOAT16>(input,
                                                      output_indices,
                                                      offsets,
                                                      lengths,
                                                      num_rows,
                                                      seq_len_ptr,
                                                      batch_id_per_token_ptr,
                                                      block_tables_ptr,
                                                      max_block_num,
                                                      top_k,
                                                      q_num_heads,
                                                      max_len,
                                                      row_states_ptr,
                                                      stream);
  } else if (input_dtype == paddle::DataType::FLOAT32) {
    status = DispatchTopK<paddle::DataType::FLOAT32>(input,
                                                     output_indices,
                                                     offsets,
                                                     lengths,
                                                     num_rows,
                                                     seq_len_ptr,
                                                     batch_id_per_token_ptr,
                                                     block_tables_ptr,
                                                     max_block_num,
                                                     top_k,
                                                     q_num_heads,
                                                     max_len,
                                                     row_states_ptr,
                                                     stream);
  } else {
    PD_THROW("input_dtype should be one of [bf16, float]");
  }
}

PD_BUILD_STATIC_OP(radix_topk_ragged_transform)
    .Inputs({"input",
             "output_indices",
             "offsets",
             "lengths",
             paddle::Optional("seq_len_decoder"),
             paddle::Optional("batch_id_per_token"),
             paddle::Optional("block_tables"),
             paddle::Optional("maybe_row_states_buffer")})
    .Attrs({"top_k : int", "q_num_heads : int", "max_block_num : int"})
    .SetKernelFn(PD_KERNEL(RadixTopkRaggedTransform));
