#include "paddle/extension.h"
#include "moba_kernel.hpp"
#include "moba_attention_utils.hpp"
#include "moba_attention.h"

namespace moba {

template <typename T>
std::vector<paddle::Tensor> DispatchQkSortDecoder(
        const paddle::Tensor& qk_gate_weight,
        const paddle::Tensor& seq_len_encoder,
        const paddle::Tensor& seq_len_decoder,
        const int head_num,
        const int kv_head_num,
        const int top_k_left,
        const int top_k_right,
        const int use_moba_seq_limit) {

    constexpr int kMobaBlockSize = 128;
    constexpr int kMaxN = 1024;

    const int batch_size = seq_len_decoder.dims()[0];
    paddle::Tensor qk_gate_topk_idx = paddle::empty({batch_size, kv_head_num, kMaxN}, paddle::DataType::INT32, qk_gate_weight.place());

    moba::qk_gate_sort_decoder<kMaxN, kMobaBlockSize, T>(
        qk_gate_weight.data<T>(),
        qk_gate_topk_idx.data<int>(),
        seq_len_decoder.data<int>(),
        head_num,
        kv_head_num,
        batch_size,
        top_k_left,
        top_k_right,
        use_moba_seq_limit,
        qk_gate_weight.stream()
    );

    return {qk_gate_topk_idx};
}

std::vector<paddle::Tensor> QkSortDecoder(
        const paddle::Tensor& qk_gate_weight,
        const paddle::Tensor& seq_len_encoder,
        const paddle::Tensor& seq_len_decoder,
        const int head_num,
        const int kv_head_num,
        const int top_k_left,
        const int top_k_right,
        const int use_moba_seq_limit) {

    if (qk_gate_weight.dtype() == paddle::DataType::FLOAT16) {
        return std::move(
            DispatchQkSortDecoder<phi::dtype::float16>(
                qk_gate_weight,
                seq_len_encoder,
                seq_len_decoder,
                head_num,
                kv_head_num,
                top_k_left,
                top_k_right,
                use_moba_seq_limit)
        );
    } else if (qk_gate_weight.dtype() == paddle::DataType::BFLOAT16) {
        return std::move(
            DispatchQkSortDecoder<phi::dtype::bfloat16>(
                qk_gate_weight,
                seq_len_encoder,
                seq_len_decoder,
                head_num,
                kv_head_num,
                top_k_left,
                top_k_right,
                use_moba_seq_limit)
        );
    }

}




template <typename T>
std::vector<paddle::Tensor> DispatchQkSortEncoder(
        const paddle::Tensor& qk_gate_weight,
        const paddle::Tensor& seq_len_encoder,
        const paddle::Tensor& seq_len_decoder,
        const paddle::Tensor& cu_seq_q,
        const paddle::Tensor& cu_seq_k,
        const paddle::Tensor& cu_seq_q_pack,
        const paddle::Tensor& q_pack_tokens,
        const int max_seq_q,
        const int max_seq_k,
        const int head_num,
        const int kv_head_num,
        const int top_k_left,
        const int top_k_right,
        const int use_moba_seq_limit) {
    constexpr int kBlockM = 128;
    constexpr int kBlockN = 128;
    constexpr int kMobaBlockSize = 128;
    constexpr int kMaxN = 1024;
    using cute_type = typename moba::cuteType<T>::type;
    const int batch_size = seq_len_encoder.dims()[0];

    paddle::Tensor qk_gate_topk_idx = paddle::empty({q_pack_tokens.data<int>()[0] / kBlockM, head_num, kMaxN}, paddle::DataType::INT32, qk_gate_weight.place());

    qk_gate_sort_encoder<kBlockM, kMaxN, kMobaBlockSize, cute_type>(
            reinterpret_cast<const cute_type *>(qk_gate_weight.data<T>()),
            qk_gate_topk_idx.data<int>(),
            seq_len_encoder.data<int>(),
            seq_len_decoder.data<int>(),
            cu_seq_q.data<int>(),
            cu_seq_k.data<int>(),
            cu_seq_q_pack.data<int>(),
            use_moba_seq_limit,
            max_seq_q,
            max_seq_k,
            head_num,
            kv_head_num,
            batch_size,
            top_k_left,
            top_k_right,
            qk_gate_weight.stream());

    return {qk_gate_topk_idx};
}


std::vector<paddle::Tensor> QkSortEncoder(
        const paddle::Tensor& qk_gate_weight,
        const paddle::Tensor& seq_len_encoder,
        const paddle::Tensor& seq_len_decoder,
        const paddle::Tensor& cu_seq_q,
        const paddle::Tensor& cu_seq_k,
        const paddle::Tensor& cu_seq_q_pack,
        const paddle::Tensor& q_pack_tokens,
        const int max_seq_q,
        const int max_seq_k,
        const int head_num,
        const int kv_head_num,
        const int top_k_left,
        const int top_k_right,
        const int use_moba_seq_limit) {
    if (qk_gate_weight.dtype() == paddle::DataType::FLOAT16) {
        return std::move(
            DispatchQkSortEncoder<phi::dtype::float16>(
                qk_gate_weight,
                seq_len_encoder,
                seq_len_decoder,
                cu_seq_q,
                cu_seq_k,
                cu_seq_q_pack,
                q_pack_tokens,
                max_seq_q,
                max_seq_k,
                head_num,
                kv_head_num,
                top_k_left,
                top_k_right,
                use_moba_seq_limit
            )
        );
    } else if (qk_gate_weight.dtype() == paddle::DataType::BFLOAT16) {
        return std::move(
            DispatchQkSortEncoder<phi::dtype::bfloat16>(
                qk_gate_weight,
                seq_len_encoder,
                seq_len_decoder,
                cu_seq_q,
                cu_seq_k,
                cu_seq_q_pack,
                q_pack_tokens,
                max_seq_q,
                max_seq_k,
                head_num,
                kv_head_num,
                top_k_left,
                top_k_right,
                use_moba_seq_limit
            )
        );
    }
}

template <typename T>
std::vector<paddle::Tensor> DispatchMobaQKGemm(
        const paddle::Tensor& q_input,
        const paddle::Tensor& k_block_means,
        const paddle::Tensor& seq_len_encoder,
        const paddle::Tensor& seq_len_decoder,
        const paddle::Tensor& cu_seq_q,
        const paddle::Tensor& cu_seq_k,
        const int max_seq_q,
        const int max_seq_k,
        const int head_num,
        const int kv_head_num,
        const bool is_split_kv,
        const int use_moba_seq_limit) {

    constexpr int kMobaBlockSize = 128;
    constexpr int kMaxN = 1024;
    const int batch_size = seq_len_encoder.dims()[0];
    using cute_type = typename moba::cuteType<T>::type;
    if (is_split_kv) {
        paddle::Tensor qk_gate_weight = paddle::empty({batch_size, head_num, kMaxN}, q_input.dtype(), q_input.place());
        moba::qk_gemm<cute_type, 16, kMobaBlockSize, kMobaBlockSize, kMaxN, true>(
            reinterpret_cast<const cute_type*>(q_input.data<T>()),
            reinterpret_cast<const cute_type*>(k_block_means.data<T>()),
            reinterpret_cast<cute_type*>(qk_gate_weight.data<T>()),
            seq_len_encoder.data<int>(),
            seq_len_decoder.data<int>(),
            cu_seq_q.data<int>(),
            cu_seq_k.data<int>(),
            use_moba_seq_limit,
            max_seq_q,
            max_seq_k,
            head_num,
            kv_head_num,
            batch_size,
            q_input.stream()
        );
        return {qk_gate_weight};
    } else {
        constexpr int kBlockM = 128;
        constexpr int kBlockN = 128;
        const int token_num = q_input.dims()[0];
        paddle::Tensor qk_gate_weight = paddle::empty({token_num, head_num, kMaxN}, q_input.dtype(), q_input.place());
        qk_gemm<cute_type, kBlockM, kBlockN, kMobaBlockSize, kMaxN, false>(
            reinterpret_cast<cute_type *>(const_cast<T*>(q_input.data<T>())),
            reinterpret_cast<cute_type *>(const_cast<T*>(k_block_means.data<T>())),
            reinterpret_cast<cute_type *>(qk_gate_weight.data<T>()),
            seq_len_encoder.data<int>(),
            seq_len_decoder.data<int>(),
            cu_seq_q.data<int>(),
            cu_seq_k.data<int>(),
            use_moba_seq_limit,
            max_seq_q,
            max_seq_k,
            head_num,
            kv_head_num,
            batch_size,
            q_input.stream());
        return {qk_gate_weight};
    }
}

std::vector<paddle::Tensor> MobaQKGemm(
        const paddle::Tensor& q_input,
        const paddle::Tensor& k_block_means,
        const paddle::Tensor& seq_len_encoder,
        const paddle::Tensor& seq_len_decoder,
        const paddle::Tensor& cu_seq_q,
        const paddle::Tensor& cu_seq_k,
        const int max_seq_q,
        const int max_seq_k,
        const int head_num,
        const int kv_head_num,
        const bool is_split_kv,
        const int use_moba_seq_limit) {

    if (q_input.dtype() == paddle::DataType::FLOAT16) {
        return std::move(
            DispatchMobaQKGemm<phi::dtype::float16>(
                q_input,
                k_block_means,
                seq_len_encoder,
                seq_len_decoder,
                cu_seq_q,
                cu_seq_k,
                max_seq_q,
                max_seq_k,
                head_num,
                kv_head_num,
                is_split_kv,
                use_moba_seq_limit
            )
        );
    } else if (q_input.dtype() == paddle::DataType::BFLOAT16) {
        return std::move(
            DispatchMobaQKGemm<phi::dtype::bfloat16>(
                q_input,
                k_block_means,
                seq_len_encoder,
                seq_len_decoder,
                cu_seq_q,
                cu_seq_k,
                max_seq_q,
                max_seq_k,
                head_num,
                kv_head_num,
                is_split_kv,
                use_moba_seq_limit
            )
        );
    }
}

};

PD_BUILD_OP(moba_qk_gemm)
    .Inputs({
        "q_input",
        "k_block_means",
        "seq_len_encoder",
        "seq_len_decoder",
        "cu_seq_q",
        "cu_seq_k",
        "max_seq_q",
        "max_seq_k"})
    .Attrs({
        "head_num: int",
        "kv_head_num: int",
        "is_split_kv: bool",
        "use_moba_seq_limit: int"})
    .Outputs({"qk_gate_weight"})
    .SetKernelFn(PD_KERNEL(moba::MobaQKGemm));

PD_BUILD_OP(moba_qk_sort_encoder)
    .Inputs({
        "qk_gate_weight",
        "seq_len_encoder",
        "seq_len_decoder",
        "cu_seq_q",
        "cu_seq_k",
        "cu_seq_q_pack",
        "q_pack_tokens",
        "max_seq_q",
        "max_seq_k"})
    .Attrs({
        "head_num: int",
        "kv_head_num: int",
        "top_k_left: int",
        "top_k_right: int",
        "use_moba_seq_limit: int"})
    .Outputs({"qk_gate_topk_idx"})
    .SetKernelFn(PD_KERNEL(moba::QkSortEncoder));

PD_BUILD_OP(moba_qk_sort_decoder)
    .Inputs({
        "qk_gate_weight",
        "seq_len_encoder",
        "seq_len_decoder"})
    .Attrs({
        "head_num: int",
        "kv_head_num: int",
        "top_k_left: int",
        "top_k_right: int",
        "use_moba_seq_limit: int"})
    .Outputs({"qk_gate_topk_idx"})
    .SetKernelFn(PD_KERNEL(moba::QkSortDecoder));
