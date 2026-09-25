/******************************************************************************
 * Copyright (c) 2024, Junxian Guo.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
namespace FLASH_NAMESPACE {

class fwdIteratorBase{
};


// ////////////////////////////////////////////////////////////////////////////////////////////////////
class fwdStreaming: public fwdIteratorBase{
    public:
    template<typename Params, typename BlockInfo>
    __device__ fwdStreaming(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int n_block_min, int n_block_max) {//row first
        this -> row_factor = params.m_block_dim / kBlockM;
        this -> col_factor = params.n_block_dim / kBlockN;
        this -> sink_block_num = params.streaming_info[head_idx * 2] * col_factor;
        this -> local_block_num = params.streaming_info[head_idx * 2 + 1] * col_factor;
        this -> m_block_dim = params.m_block_dim;
        this -> n_block_dim = params.n_block_dim;
        this -> mask_type = params.head_mask_type[head_idx];
        this -> n_block_min = n_block_min;
        this -> n_block_max = n_block_max;
        int act_k = binfo.actual_seqlen_k;
        int act_q = binfo.actual_seqlen_q;
        bool causal = params.is_causal;
        if (causal){
            int start_row_idx = max(int((act_q-act_k)/m_block_dim), 0);
            this -> start_block_val = (cute::ceil_div(max(act_k - act_q, 0), n_block_dim) + 1 + loop_step_idx/row_factor - start_row_idx) * col_factor;
        }else{
            this -> start_block_val = max(cute::ceil_div(n_block_max * kBlockN, n_block_dim) * col_factor, 0);
        };
        this -> no_gap = start_block_val - n_block_min < sink_block_num + local_block_num;
        this -> max_block_idx = min(sink_block_num + local_block_num, start_block_val - n_block_min);

        assert(mask_type < 0);
        assert(params.m_block_dim % kBlockM == 0);
        assert(params.n_block_dim % kBlockN == 0);
    };

    __device__ int mask_val(int block_col_idx) const {
        if (block_col_idx > max_block_idx || block_col_idx < 0){
            return -1;
        };
        int ret = 0;
        if (no_gap){
            ret = start_block_val - 1 - block_col_idx;
            return ret >= n_block_min ? ret : -1;
        }else{
            if (block_col_idx < local_block_num){
                return start_block_val - 1 - block_col_idx;
            }else{
                ret = sink_block_num - 1 - (block_col_idx - local_block_num);
                return ret >= n_block_min ? ret : -1;
            };
        };
    };

    __device__ int max_no_larger(int target) const {
        if(max_block_idx == 0){
            return -1;
        };
        int left = 0;
        int right = max_block_idx - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (mask_val(mid) > target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            };
        };
        return (left < max_block_idx && mask_val(left) <= target) ? left : -1;
    };

    int sink_block_num, local_block_num;
    int start_block_val;
    bool no_gap;
    
    int max_block_idx;
    int m_block_dim, n_block_dim;
    int mask_type;
    int n_block_min, n_block_max;
    int row_factor, col_factor;
};


class fwdExactStreaming: public fwdIteratorBase{
    public:
    template<typename Params, typename BlockInfo>
    __device__ fwdExactStreaming(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int n_block_min, int n_block_max) {//row first
        this -> row_factor = params.m_block_dim / kBlockM;
        this -> col_factor = params.n_block_dim / kBlockN;
        int sink_num = params.streaming_info[head_idx * 2];
        int local_num = params.streaming_info[head_idx * 2 + 1];
        this -> m_block_dim = params.m_block_dim;
        this -> n_block_dim = params.n_block_dim;
        this -> sink_block_num = cute::ceil_div(sink_num, n_block_dim) * col_factor;
        this -> local_block_num = (cute::ceil_div(local_num, n_block_dim)+2) * col_factor;

        
        
        this -> mask_type = params.head_mask_type[head_idx];
        this -> n_block_min = n_block_min;
        this -> n_block_max = n_block_max;
        int act_k = binfo.actual_seqlen_k;
        int act_q = binfo.actual_seqlen_q;
        bool causal = params.is_causal;
        if (causal){
            int start_row_idx = max(int((act_q-act_k)/m_block_dim), 0);
            this -> start_block_val = (cute::ceil_div(max(act_k - act_q, 0), n_block_dim) + 1 + loop_step_idx/row_factor - start_row_idx) * col_factor;
        }else{
            this -> start_block_val = max(cute::ceil_div(n_block_max * kBlockN, n_block_dim) * col_factor, 0);
        };
        this -> no_gap = start_block_val - n_block_min < sink_block_num + local_block_num;
        this -> max_block_idx = min(sink_block_num + local_block_num, start_block_val - n_block_min);

        assert(mask_type < 0);
        assert(params.m_block_dim % kBlockM == 0);
        assert(params.n_block_dim % kBlockN == 0);
    };

    __device__ int mask_val(int block_col_idx) const {
        if (block_col_idx > max_block_idx || block_col_idx < 0){
            return -1;
        };
        int ret = 0;
        if (no_gap){
            ret = start_block_val - 1 - block_col_idx;
            return ret >= n_block_min ? ret : -1;
        }else{
            if (block_col_idx < local_block_num){
                return start_block_val - 1 - block_col_idx;
            }else{
                ret = sink_block_num - 1 - (block_col_idx - local_block_num);
                return ret >= n_block_min ? ret : -1;
            };
        };
    };

    __device__ int max_no_larger(int target) const {
        if(max_block_idx == 0){
            return -1;
        };
        int left = 0;
        int right = max_block_idx - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (mask_val(mid) > target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            };
        };
        return (left < max_block_idx && mask_val(left) <= target) ? left : -1;
    };

    int sink_block_num, local_block_num;
    int start_block_val;
    bool no_gap;
    
    int max_block_idx;
    int m_block_dim, n_block_dim;
    int mask_type;
    int n_block_min, n_block_max;
    int row_factor, col_factor;
};

// ////////////////////////////////////////////////////////////////////////////////////////////////////

class fwdBlockmask: public fwdIteratorBase{
    public:
    template<typename Params, typename BlockInfo>
    __device__ fwdBlockmask(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int n_block_min, int n_block_max) {//row first
        this -> row_factor = params.m_block_dim / kBlockM;
        this -> col_factor = params.n_block_dim / kBlockN;
        this -> max_block_idx = cute::ceil_div(binfo.actual_seqlen_k, params.n_block_dim) * col_factor;
        this -> m_block_dim = params.m_block_dim;
        this -> n_block_dim = params.n_block_dim;
        this -> mask_type = params.head_mask_type[head_idx];
        this -> n_block_min = n_block_min;
        this -> n_block_max = n_block_max;

        assert(mask_type > 0);
        assert(params.m_block_dim % kBlockM == 0);
        assert(params.n_block_dim % kBlockN == 0);
        
        blockmask_ptr = params.blockmask + (batch_idx * params.num_blocksparse_heads + mask_type - 1) * int(params.seqlen_q_rounded / m_block_dim) * int(params.seqlen_k_rounded / n_block_dim) + int(loop_step_idx / row_factor) * int(params.seqlen_k_rounded / n_block_dim);
    };

    __device__ int mask_val(int block_col_idx) const {
        if (block_col_idx > max_block_idx || block_col_idx < 0){
            return -1;
        };
        int real_block_idx = block_col_idx / col_factor;
        int block_col_offset = block_col_idx % col_factor;
        int mask_val = blockmask_ptr[real_block_idx];
        return mask_val == -1 ? -1 : col_factor * mask_val + col_factor - 1 - block_col_offset;
    };

    __device__ int max_no_larger(int target) const {
        if(max_block_idx == 0){
            return -1;
        };
        int left = 0;
        int right = max_block_idx - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (mask_val(mid) > target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            };
        };
        return (left < max_block_idx && mask_val(left) <= target) ? left : -1;
    };

    int *blockmask_ptr;
    int max_block_idx;
    int m_block_dim, n_block_dim;
    int mask_type;
    int n_block_min, n_block_max;
    int row_factor, col_factor;
};

// ////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool Is_streaming, bool Is_exact_streaming>   
class fwdIterator{};

template<>
struct fwdIterator<false, false>: public fwdBlockmask{
    template<typename Params, typename BlockInfo>
    __device__ fwdIterator(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int n_block_min, int n_block_max): fwdBlockmask(params, binfo, kBlockM, kBlockN, batch_idx, head_idx, loop_step_idx, n_block_min, n_block_max) {};
};

template<>
struct fwdIterator<true, false>: public fwdStreaming{
    template<typename Params, typename BlockInfo>
    __device__ fwdIterator(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int n_block_min, int n_block_max): fwdStreaming(params, binfo, kBlockM, kBlockN, batch_idx, head_idx, loop_step_idx, n_block_min, n_block_max) {};
};

template<>
struct fwdIterator<true, true>: public fwdExactStreaming{
    template<typename Params, typename BlockInfo>
    __device__ fwdIterator(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int n_block_min, int n_block_max): fwdExactStreaming(params, binfo, kBlockM, kBlockN, batch_idx, head_idx, loop_step_idx, n_block_min, n_block_max) {};
};

////////////////////////////////////////////////////////////////////////////////////////////////////

class bwdIteratorBase{
};


struct bwdStreaming: public bwdIteratorBase{
    public:
    template<typename Params, typename BlockInfo>
    __device__ bwdStreaming(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int m_block_min, int m_block_max) {// col first
        this -> row_factor = params.m_block_dim / kBlockM;
        this -> col_factor = params.n_block_dim / kBlockN;
        
        this -> m_block_dim = params.m_block_dim;
        this -> n_block_dim = params.n_block_dim;
        this -> mask_type = params.head_mask_type[head_idx];
        this -> m_block_min = m_block_min;
        this -> m_block_max = m_block_max;

        int mask_block_col = cute::ceil_div(loop_step_idx+1, col_factor);
        int sink = (this -> mask_type) < 0 ? params.streaming_info[head_idx * 2]: cute::ceil_div(binfo.actual_seqlen_k, this -> n_block_dim);
        int local = (this -> mask_type) < 0 ? params.streaming_info[head_idx * 2 + 1]: 0;
        this -> sink_block_num = sink * col_factor;
        this -> local_block_num = local * col_factor;
        int act_q = binfo.actual_seqlen_q;
        int act_k = binfo.actual_seqlen_k;
        bool causal = params.is_causal;

        if(mask_block_col <= sink){
            this -> start_block_val = m_block_max;
            this -> max_block_idx = m_block_max - m_block_min;
        }else{
            if (causal){
                int free_token_num = act_q - min(act_q, act_k - loop_step_idx * kBlockN);
                int end_mask_block_row_idx = free_token_num / params.m_block_dim;//zero based
                int num_mask_block_in_end_row = max(0, cute::ceil_div(act_k - act_q + (end_mask_block_row_idx + 1) * params.m_block_dim, params.n_block_dim));
                int local_col_mask_block_num = max(0, local - (num_mask_block_in_end_row - mask_block_col));
                if(local_col_mask_block_num > 0){
                    this -> start_block_val = min((end_mask_block_row_idx + local_col_mask_block_num) * row_factor, m_block_max);
                    this -> max_block_idx = min(local_col_mask_block_num * row_factor, m_block_max - m_block_min);
                }else{
                    this -> start_block_val = 0;
                    this -> max_block_idx = 0;
                };
            }else{
                int n_mask_block_col = max(cute::ceil_div(act_k, n_block_dim), 0);
                bool in_none_causal_local = !causal && mask_block_col <= n_mask_block_col && mask_block_col > n_mask_block_col - local;
                if(in_none_causal_local){
                    this -> start_block_val = m_block_max;
                    this -> max_block_idx = m_block_max - m_block_min;
                }else{
                    this -> start_block_val = 0;
                    this -> max_block_idx = 0;
                };
            };
        }
        
        assert(mask_type <= 0); //for blocksparse, mask_type > 0; for streaming, mask_type < 0; for dense, mask_type = 0
        assert(params.m_block_dim % kBlockM == 0);
        assert(params.n_block_dim % kBlockN == 0);
    };

    __device__ int mask_val(int block_row_idx) const {
        if (block_row_idx > max_block_idx || block_row_idx < 0){
            return -1;
        };
        int ret = start_block_val - 1 - block_row_idx;
        return ret >= m_block_min ? ret : -1;
    };

    __device__ int max_no_larger(int target) const {
        if(max_block_idx == 0){
            return -1;
        };
        int left = 0;
        int right = max_block_idx - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (mask_val(mid) > target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            };
        };
        return (left < max_block_idx && mask_val(left) <= target) ? left : -1;
    };

    int sink_block_num, local_block_num;
    int start_block_val;

    int max_block_idx;
    int m_block_dim, n_block_dim;
    int mask_type;
    int m_block_min, m_block_max;
    int row_factor, col_factor;
};

struct bwdBlockmask: public bwdIteratorBase{
    public:
    template<typename Params, typename BlockInfo>
    __device__ bwdBlockmask(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int m_block_min, int m_block_max) {
        this -> row_factor = params.m_block_dim / kBlockM;
        this -> col_factor = params.n_block_dim / kBlockN;
        this -> max_block_idx = cute::ceil_div(binfo.actual_seqlen_q, params.m_block_dim) * row_factor;
        this -> m_block_dim = params.m_block_dim;
        this -> n_block_dim = params.n_block_dim;
        this -> mask_type = params.head_mask_type[head_idx];
        this -> m_block_min = m_block_min;
        this -> m_block_max = m_block_max;
        assert(mask_type > 0);
        assert(params.m_block_dim % kBlockM == 0);
        assert(params.n_block_dim % kBlockN == 0);

        blockmask_ptr = params.blockmask + (batch_idx * params.num_blocksparse_heads + mask_type - 1) * int(params.seqlen_k_rounded / n_block_dim) * int(params.seqlen_q_rounded / m_block_dim) + int(loop_step_idx / col_factor) * int(params.seqlen_q_rounded / m_block_dim);
    };

    __device__ int mask_val(int block_row_idx) const {
        if (block_row_idx > max_block_idx || block_row_idx < 0){
            return -1;
        };
        int real_block_idx = block_row_idx / row_factor;
        int block_row_offset = block_row_idx % row_factor;
        int mask_val = blockmask_ptr[real_block_idx];
        return mask_val == -1 ? -1 : row_factor * mask_val + row_factor - 1 - block_row_offset;
    };

    __device__ int max_no_larger(int target) const {
        if(max_block_idx == 0){
            return -1;
        };
        int left = 0;
        int right = max_block_idx - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (mask_val(mid) > target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            };
        };
        return (left < max_block_idx && mask_val(left) <= target) ? left : -1;
    };

    int *blockmask_ptr;
    int max_block_idx;
    int m_block_dim, n_block_dim;
    int mask_type;
    int m_block_min, m_block_max;
    int row_factor, col_factor;
};



template<bool Is_streaming>   
class bwdIterator{};

template<>
struct bwdIterator<false>: public bwdBlockmask{
    template<typename Params, typename BlockInfo>
    __device__ bwdIterator(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int m_block_min, int m_block_max): bwdBlockmask(params, binfo, kBlockM, kBlockN, batch_idx, head_idx, loop_step_idx, m_block_min, m_block_max) {};
};

template<>
struct bwdIterator<true>: public bwdStreaming{
    template<typename Params, typename BlockInfo>
    __device__ bwdIterator(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int m_block_min, int m_block_max): bwdStreaming(params, binfo, kBlockM, kBlockN, batch_idx, head_idx, loop_step_idx, m_block_min, m_block_max) {};
};


////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace FLASH_NAMESPACE