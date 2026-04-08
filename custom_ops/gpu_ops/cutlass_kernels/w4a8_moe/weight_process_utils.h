/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once
#include <iostream>

void row_major_to_column_major(int8_t* col_major_tensor,
                               const int8_t* row_major_tensor,
                               const std::vector<size_t>& shape){
    size_t m = shape[0];
    size_t n = shape[1];
    for(auto i=0;i<m*n;i++){
        size_t im = i / n;
        size_t in = i % n;
        col_major_tensor[in*m+im] = row_major_tensor[im*n+in];
    }
}

void add_bias_and_interleave_int8s_inplace(int8_t* int8_tensor_ptr,
                                           int64_t num_elts)
{
    int8_t* int8_tensor = reinterpret_cast<int8_t *>(int8_tensor_ptr);
    for (int ii = 0; ii < num_elts; ++ii) {
        int8_tensor[ii] = int8_t(int(int8_tensor[ii]) + 128);
        // int8_tensor[ii] = int8_t(int(int8_tensor[ii]));
    }

    // Step 2 will transform the layout of a 32-bit register in CUDA in order to match the int4 layout. This has no
    // performance benefit and is purely so that int4 and int8 have the same layout.
    // Pictorially, this does the following:
    // bit 32                                                      0
    //      [elt_3  elt_2  elt_1  elt_0] (each elt occupies 8 bits)
    //
    // And it will rearrange the output 32 bit register to be the following:
    // bit 32                                                      0
    //      [elt_3  elt_1  elt_2  elt_0] (each elt occupies 8 bits)

    for (int64_t base = 0; base < num_elts; base += 4) {
        std::swap(int8_tensor[base + 1], int8_tensor[base + 2]);
    }
}


void subbyte_transpose_impl_int4(int8_t*                    transposed_quantized_tensor,
                            const int8_t*              quantized_tensor,
                            const std::vector<size_t>& shape)
{
    const int bits_per_elt = 4;

    const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
    const size_t num_rows    = shape.size() == 2 ? shape[0] : shape[1];
    const size_t num_cols    = shape.size() == 2 ? shape[1] : shape[2];

    const size_t col_bytes       = num_cols * bits_per_elt / 8;
    const size_t col_bytes_trans = num_rows * bits_per_elt / 8;
    const size_t num_bytes       = size_t(num_experts) * num_rows * col_bytes;

    const uint8_t* input_byte_ptr  = reinterpret_cast<const uint8_t*>(quantized_tensor);
    uint8_t*       output_byte_ptr = reinterpret_cast<uint8_t*>(transposed_quantized_tensor);

    // static_assert(quant_type == QuantType::INT8_WEIGHT_ONLY || quant_type == QuantType::PACKED_INT4_WEIGHT_ONLY, "");
    static constexpr int ELTS_PER_BYTE = 2;

    static constexpr int M_TILE_L1 = 64;
    static constexpr int N_TILE_L1 = M_TILE_L1 / ELTS_PER_BYTE;
    uint8_t              cache_buf[M_TILE_L1][N_TILE_L1];

    static constexpr int VECTOR_WIDTH = std::min(32, N_TILE_L1);

    // We assume the dims are a multiple of vector width. Our kernels only handle dims which are multiples
    // of 64 for weight-only quantization. As a result, this seemed like a reasonable tradeoff because it
    // allows GCC to emit vector instructions.

    const int num_m_tiles = (num_rows + M_TILE_L1 - 1) / M_TILE_L1;
    const int num_n_tiles = (col_bytes + N_TILE_L1 - 1) / N_TILE_L1;

    for (size_t expert = 0; expert < num_experts; ++expert) {
        const size_t matrix_offset = expert * num_rows * col_bytes;
        for (size_t row_tile_start = 0; row_tile_start < num_rows; row_tile_start += M_TILE_L1) {
            for (size_t col_tile_start_byte = 0; col_tile_start_byte < col_bytes; col_tile_start_byte += N_TILE_L1) {

                const int row_limit = std::min(row_tile_start + M_TILE_L1, num_rows);
                const int col_limit = std::min(col_tile_start_byte + N_TILE_L1, col_bytes);

                for (int ii = 0; ii < M_TILE_L1; ++ii) {
                    const int row = row_tile_start + ii;

                    for (int jj = 0; jj < N_TILE_L1; jj += VECTOR_WIDTH) {
                        const int col = col_tile_start_byte + jj;

                        const size_t logical_src_offset = matrix_offset + row * col_bytes + col;

                        if (row < row_limit && col < col_limit) {
                            for (int v = 0; v < VECTOR_WIDTH; ++v) {
                                cache_buf[ii][jj + v] = input_byte_ptr[logical_src_offset + v];
                            }
                        }
                    }
                }


                for (int ii = 0; ii < M_TILE_L1; ++ii) {
                    // Using M_TILE_L1 here is deliberate since we assume that the cache tile
                    // is square in the number of elements (not necessarily the number of bytes).
                    for (int jj = ii + 1; jj < M_TILE_L1; ++jj) {
                        const int ii_byte       = ii / ELTS_PER_BYTE;
                        const int ii_bit_offset = ii % ELTS_PER_BYTE;

                        const int jj_byte       = jj / ELTS_PER_BYTE;
                        const int jj_bit_offset = jj % ELTS_PER_BYTE;

                        uint8_t src_elt = 0xF & (cache_buf[ii][jj_byte] >> (4 * jj_bit_offset));
                        uint8_t tgt_elt = 0xF & (cache_buf[jj][ii_byte] >> (4 * ii_bit_offset));

                        cache_buf[ii][jj_byte] &= (0xF0 >> (4 * jj_bit_offset));
                        cache_buf[jj][ii_byte] &= (0xF0 >> (4 * ii_bit_offset));

                        cache_buf[ii][jj_byte] |= (tgt_elt << (4 * jj_bit_offset));
                        cache_buf[jj][ii_byte] |= (src_elt << (4 * ii_bit_offset));
                    }
                }


                const size_t row_tile_start_trans      = col_tile_start_byte * ELTS_PER_BYTE;
                const size_t col_tile_start_byte_trans = row_tile_start / ELTS_PER_BYTE;

                const int row_limit_trans = std::min(row_tile_start_trans + M_TILE_L1, num_cols);
                const int col_limit_trans = std::min(col_tile_start_byte_trans + N_TILE_L1, col_bytes_trans);

                for (int ii = 0; ii < M_TILE_L1; ++ii) {
                    const int row = row_tile_start_trans + ii;
                    for (int jj = 0; jj < N_TILE_L1; jj += VECTOR_WIDTH) {
                        const int col = col_tile_start_byte_trans + jj;

                        const size_t logical_tgt_offset = matrix_offset + row * col_bytes_trans + col;

                        if (row < row_limit_trans && col < col_limit_trans) {
                            for (int v = 0; v < VECTOR_WIDTH; ++v) {
                                output_byte_ptr[logical_tgt_offset + v] = cache_buf[ii][jj + v];
                            }
                        }
                    }
                }
            }
        }
    }
}


void add_bias_and_interleave_int4s_inplace(int8_t* packed_int4_tensor, const size_t num_elts)
{
    const int num_bytes = num_elts / 2;

    // Step 1 will be to transform all the int4s to unsigned in order to make the dequantize take as little
    // instructions as possible in the CUDA code.
    for (size_t ii = 0; ii < num_bytes; ++ii) {
        int8_t transformed_packed_int4s = 0;
        // We don't need to mask in these ops since everything should be in the range 0-15
        int8_t transformed_first_elt = (packed_int4_tensor[ii] & 0x0F);
        int8_t transformed_second_elt = (packed_int4_tensor[ii] >> 4);

        transformed_packed_int4s |= transformed_first_elt;
        transformed_packed_int4s |= (transformed_second_elt << 4);
        packed_int4_tensor[ii] = transformed_packed_int4s;
    }

    // Step 2 will transform the layout of a 32-bit register in CUDA in order to minimize the number of shift & logical
    // instructions That are needed to extract the int4s in the GEMM main loop. Pictorially, the loop below will do the
    // following: Take as input a 32 bit register with layout: bit 32 0
    //      [elt_7  elt_6  elt_5  elt_4  elt_3  elt_2  elt_1  elt_0] (each elt occupies 4 bits)
    //
    // And it will rearrange the output 32 bit register to be the following:
    // bit 32                                                      0
    //      [elt_7  elt_5  elt_3  elt_1  elt_6  elt_4  elt_2  elt_0] (each elt occupies 4 bits)

    // FT_CHECK_WITH_INFO(num_bytes % 4 == 0, "Dimensions of int4 tensor must be a multiple of 8 for register relayout");
    const size_t num_registers = num_bytes / 4;

    uint32_t* register_ptr = reinterpret_cast<uint32_t*>(packed_int4_tensor);
    for (size_t ii = 0; ii < num_registers; ++ii) {
        const uint32_t current_register     = register_ptr[ii];
        uint32_t       transformed_register = 0;

        for (int dest_idx = 0; dest_idx < 8; ++dest_idx) {
            const int src_idx    = dest_idx < 4 ? 2 * dest_idx : 2 * (dest_idx - 4) + 1;
            const int src_shift  = 4 * src_idx;
            const int dest_shift = 4 * dest_idx;

            const uint32_t src_bits = (current_register >> src_shift) & 0xF;
            transformed_register |= (src_bits << dest_shift);

        }
        register_ptr[ii] = transformed_register;
    }
}

void permute_B_rows_for_mixed_and_int8_gemm(int8_t*                    permuted_quantized_tensor,
                                            const int8_t*              quantized_tensor,
                                            const std::vector<size_t>& shape,
                                            const int64_t              arch_version)
{

    // We only want to run this step for weight only quant.
    const size_t num_rows    = shape.size() == 2 ? shape[0] : shape[1];
    const size_t num_cols    = shape.size() == 2 ? shape[1] : shape[2];

    const int BITS_PER_ELT  = 8;
    const int K             = 16 / BITS_PER_ELT;
    const int ELTS_PER_BYTE = 8 / BITS_PER_ELT;
    const int ELTS_PER_REG  = 32 / BITS_PER_ELT;

    const uint32_t* input_byte_ptr  = reinterpret_cast<const uint32_t*>(quantized_tensor);
    uint32_t*       output_byte_ptr = reinterpret_cast<uint32_t*>(permuted_quantized_tensor);

    int       MMA_SHAPE_N    = 8;
    int       B_ROWS_PER_MMA = 8 * K;
    const int elts_in_int32  = 32 / BITS_PER_ELT;

    const int num_vec_cols = num_cols / elts_in_int32;

    // The code is written as below so it works for both int8 and packed int4.
    for (int base_row = 0; base_row < num_rows; base_row += B_ROWS_PER_MMA) {
        for (int tile_row = 0; tile_row < B_ROWS_PER_MMA; ++tile_row) {

            for (int write_col = 0; write_col < num_vec_cols; ++write_col) {
                const int write_row = base_row + tile_row;
                const int tile_read_row =
                    4 * (((tile_row % ELTS_PER_REG) / 2)) + tile_row % 2 + 2 * (tile_row / ELTS_PER_REG);

                const int read_row = base_row + tile_read_row;
                const int read_col = write_col;

                const int64_t read_offset  = int64_t(read_row) * num_vec_cols + read_col;
                const int64_t write_offset = int64_t(write_row) * num_vec_cols + write_col;

                output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
            }
        }
    }
}

// Permutes the rows of B for Turing and Ampere. Throws an error for other architectures.
// The data is permuted such that:
// For int8, each group of 16 rows is permuted using the map below:
//  0 1 8 9 2 3 10 11 4 5 12 13 6 7 14 15
//  0 1 2 3 4 5 6 7
template<int bits=8>
void permute_B_rows_for_mixed_gemm(int8_t*                    permuted_quantized_tensor,
                                   const int8_t*              quantized_tensor,
                                   const std::vector<size_t>& shape,
                                   const int64_t              arch_version)
{

    // We only want to run this step for weight only quant.
    const size_t num_rows    = shape.size() == 2 ? shape[0] : shape[1];
    const size_t num_cols    = shape.size() == 2 ? shape[1] : shape[2];

    const int BITS_PER_ELT  = bits;
    const int K             = 16 / BITS_PER_ELT;
    const int ELTS_PER_BYTE = 8 / BITS_PER_ELT;
    const int ELTS_PER_REG  = 32 / BITS_PER_ELT;

    const uint32_t* input_byte_ptr  = reinterpret_cast<const uint32_t*>(quantized_tensor);
    uint32_t*       output_byte_ptr = reinterpret_cast<uint32_t*>(permuted_quantized_tensor);

    int       MMA_SHAPE_N    = 8;
    int       B_ROWS_PER_MMA = 8 * K;
    const int elts_in_int32  = 32 / BITS_PER_ELT;

    const int num_vec_cols = num_cols / elts_in_int32;

    // The code is written as below so it works for both int8 and packed int4.
    for (int base_row = 0; base_row < num_rows; base_row += B_ROWS_PER_MMA) {
        for (int tile_row = 0; tile_row < B_ROWS_PER_MMA; ++tile_row) {

            for (int write_col = 0; write_col < num_vec_cols; ++write_col) {
                const int write_row = base_row + tile_row;
                const int tile_read_row =
                    8 * (((tile_row % ELTS_PER_REG) / 2)) + tile_row % 2 + 2 * (tile_row / ELTS_PER_REG);
                if(base_row == 0 && write_col == 0){
                    std::cout<<"tile_read_row:"<<tile_read_row<<std::endl;
                }
                const int read_row = base_row + tile_read_row;
                const int read_col = write_col;

                const int64_t read_offset  = int64_t(read_row) * num_vec_cols + read_col;
                const int64_t write_offset = int64_t(write_row) * num_vec_cols + write_col;

                output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
            }
        }
    }
}

template<int bits=4>
void permute_B_rows_for_mixed_gemm_int4(int8_t*                    permuted_quantized_tensor,
                                   const int8_t*              quantized_tensor,
                                   const std::vector<size_t>& shape,
                                   const int64_t              arch_version)
{

    // We only want to run this step for weight only quant.
    const size_t num_rows    = shape.size() == 2 ? shape[0] : shape[1];
    const size_t num_cols    = shape.size() == 2 ? shape[1] : shape[2];

    const int BITS_PER_ELT  = bits; //4
    const int K             = 16 / BITS_PER_ELT; // 4
    const int ELTS_PER_BYTE = 8 / BITS_PER_ELT; // 2
    const int ELTS_PER_REG  = 32 / BITS_PER_ELT; // 8

    const uint32_t* input_byte_ptr  = reinterpret_cast<const uint32_t*>(quantized_tensor);
    uint32_t*       output_byte_ptr = reinterpret_cast<uint32_t*>(permuted_quantized_tensor);

    int       MMA_SHAPE_N    = 8;
    int       B_ROWS_PER_MMA = 8 * K; // 32
    const int elts_in_int32  = 32 / BITS_PER_ELT;

    const int num_vec_cols = num_cols / elts_in_int32;
    const std::vector<int> tile_col_map{
                                       0,2,16,18,
                                       1,3,17,19,
                                       4,6,20,22,
                                       5,7,21,23,
                                       8,10,24,26,
                                       9,11,25,27,
                                       12,14,28,30,
                                       13,15,29,31};

    // const std::vector<int> tile_col_map{
    //                   0                   0,2,16,18,
    //                   4                   1,3,17,19,
    //                   8                   4,6,20,22,
    //                   12                  5,7,21,23,
    //                   16                  8,10,24,26,
    //                   20                  9,11,25,27,
    //                   24                  12,14,28,30,
    //                   28                  13,15,29,31};
    // std::vector<int> tile_col_map(32);
    // for(int i=0;i<32;i++){
    //     tile_col_map[i]=i;
    // }
    // // tile_col_map[1]=4;
    // tile_col_map[0]=0;
    // tile_col_map[4]=1;
    // tile_col_map[1]=2;
    // tile_col_map[5]=3;
    // tile_col_map[8]=4;
    // tile_col_map[12]=5;
    // tile_col_map[9]=6;
    // tile_col_map[13]=7;
    // tile_col_map[16]=8;
    // tile_col_map[20]=9;
    // tile_col_map[17]=10;
    // tile_col_map[21]=11;
    // tile_col_map[24]=12;
    // tile_col_map[28]=13;
    // tile_col_map[25]=14;
    // tile_col_map[29]=15;

    // tile_col_map[4]=1;
    // tile_col_map[4]=1;
    // tile_col_map[4]=2;

    // The code is written as below so it works for both int8 and packed int4.
    for (int base_row = 0; base_row < num_rows; base_row += B_ROWS_PER_MMA) {
        for (int tile_row = 0; tile_row < B_ROWS_PER_MMA; ++tile_row) {

            for (int write_col = 0; write_col < num_vec_cols; ++write_col) {
                const int write_row = base_row + tile_row;
                // const int tile_read_row =
                //     8 * (((tile_row % ELTS_PER_REG) / 2)) + tile_row % 2 + 2 * (tile_row / ELTS_PER_REG);
                // const int tile_read_row = std::distance(tile_col_map.begin(), std::find(tile_col_map.begin(),tile_col_map.end(), tile_row));
                const int tile_read_row = tile_col_map[tile_row];
                if(base_row == 0 && write_col == 0){
                    std::cout<<" write_row:"<<tile_row<<" tile_read_row:"<<tile_read_row<<std::endl;
                }
                const int read_row = base_row + tile_read_row;
                const int read_col = write_col;

                const int64_t read_offset  = int64_t(read_row) * num_vec_cols + read_col;
                const int64_t write_offset = int64_t(write_row) * num_vec_cols + write_col;

                output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
            }
        }
    }
}


void interleave_column_major_tensor(int8_t*                    interleaved_quantized_tensor,
                                    const int8_t*              quantized_tensor,
                                    const std::vector<size_t>& shape)
{

    // We only want to run this step for weight only quant.
    std::cout<<"### in interleave_column_major_tensor"<<std::endl;
    const size_t num_rows    = shape.size() == 2 ? shape[0] : shape[1];
    const size_t num_cols    = shape.size() == 2 ? shape[1] : shape[2];

    const size_t BITS_PER_ELT  = 8;
    const size_t elts_in_int32 = 32 / BITS_PER_ELT;

    const size_t rows_per_tile = 64;
    std::cout<<"running interleave_column_major_tensor"<<std::endl;
    std::cout<<"num_rows:"<<num_rows<<","
             <<"num_cols:"<<num_cols<<","
             <<"BITS_PER_ELT:"<<BITS_PER_ELT<<","
             <<"elts_in_int32:"<<elts_in_int32<<","
             <<"rows_per_tile:"<<rows_per_tile<<std::endl;

    const uint32_t* input_byte_ptr  = reinterpret_cast<const uint32_t*>(quantized_tensor);
    uint32_t*       output_byte_ptr = reinterpret_cast<uint32_t*>(interleaved_quantized_tensor);


    const size_t num_vec_rows      = num_rows / elts_in_int32;
    const size_t vec_rows_per_tile = rows_per_tile / elts_in_int32;
    const size_t interleave        = 2;
    std::cout<<"num_vec_rows:"<<num_vec_rows<<","
             <<"vec_rows_per_tile:"<<vec_rows_per_tile<<","
             <<"interleave:"<<interleave<<std::endl;
    for (int read_col = 0; read_col < num_cols; ++read_col) {
        const size_t write_col = read_col / interleave;
        for (int base_vec_row = 0; base_vec_row < num_vec_rows; base_vec_row += vec_rows_per_tile) {
            for (int vec_read_row = base_vec_row;
                    vec_read_row < std::min(num_vec_rows, base_vec_row + vec_rows_per_tile);
                    ++vec_read_row) {
                const size_t vec_write_row = interleave * base_vec_row
                                                + vec_rows_per_tile * (read_col % interleave)
                                                + vec_read_row % vec_rows_per_tile;

                const size_t read_offset = size_t(read_col) * num_vec_rows + vec_read_row;
                const size_t write_offset = size_t(write_col) * num_vec_rows * interleave + vec_write_row;
                output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
            }
        }
    }
}


void interleave_column_major_tensor_int4(int8_t*                    interleaved_quantized_tensor,
                                    const int8_t*              quantized_tensor,
                                    const std::vector<size_t>& shape)
{

    // We only want to run this step for weight only quant.
    std::cout<<"### in interleave_column_major_tensor"<<std::endl;
    const size_t num_rows    = shape.size() == 2 ? shape[0] : shape[1];
    const size_t num_cols    = shape.size() == 2 ? shape[1] : shape[2];

    const size_t BITS_PER_ELT  = 4;
    const size_t elts_in_int32 = 32 / BITS_PER_ELT;

    const size_t rows_per_tile = 64;
    std::cout<<"running interleave_column_major_tensor"<<std::endl;
    std::cout<<"num_rows:"<<num_rows<<","
             <<"num_cols:"<<num_cols<<","
             <<"BITS_PER_ELT:"<<BITS_PER_ELT<<","
             <<"elts_in_int32:"<<elts_in_int32<<","
             <<"rows_per_tile:"<<rows_per_tile<<std::endl;

    const uint32_t* input_byte_ptr  = reinterpret_cast<const uint32_t*>(quantized_tensor);
    uint32_t*       output_byte_ptr = reinterpret_cast<uint32_t*>(interleaved_quantized_tensor);


    const size_t num_vec_rows      = num_rows / elts_in_int32;
    const size_t vec_rows_per_tile = rows_per_tile / elts_in_int32;
    const size_t interleave        = 4;
    std::cout<<"num_vec_rows:"<<num_vec_rows<<","
             <<"vec_rows_per_tile:"<<vec_rows_per_tile<<","
             <<"interleave:"<<interleave<<std::endl;
    for (int read_col = 0; read_col < num_cols; ++read_col) {
        const size_t write_col = read_col / interleave;
        for (int base_vec_row = 0; base_vec_row < num_vec_rows; base_vec_row += vec_rows_per_tile) {
            for (int vec_read_row = base_vec_row;
                    vec_read_row < std::min(num_vec_rows, base_vec_row + vec_rows_per_tile);
                    ++vec_read_row) {
                const size_t vec_write_row = interleave * base_vec_row
                                                + vec_rows_per_tile * (read_col % interleave)
                                                + vec_read_row % vec_rows_per_tile;

                const size_t read_offset = size_t(read_col) * num_vec_rows + vec_read_row;
                const size_t write_offset = size_t(write_col) * num_vec_rows * interleave + vec_write_row;
                output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
            }
        }
    }
}
