// Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
* @file matrix_multiply.cpp
* @author tianjinjin@baidu.com
* @date Mon Mar  8 11:36:11 CST 2021
* @brief 
**/

#include "poros/converter/gpu/converter_util.h"
#include "poros/converter/gpu/matrix_multiply.h"
#include "poros/converter/gpu/weight.h"
#include "poros/engine/tensorrt_engine.h"
#include "poros/engine/trtengine_util.h"
#include "poros/context/poros_global.h"
#include "poros/util/macros.h"
#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

/** aten::matmul(Tensor self, Tensor other) -> Tensor 
 * this can be much complicated than i expected.
 * The behavior depends on the dimensionality of the Tensors as follows:
- If both Tensors are 1-dimensional, the dot product (scalar) is returned.
- If both arguments are 2-dimensional, the matrix-matrix product is returned.
- If the first argument is 1-dimensional and the second argument is 2-dimensional,
  a 1 is prepended to its dimension for the purpose of the matrix multiply.
  After the matrix multiply, the prepended dimension is removed.
- If the first argument is 2-dimensional and the second argument is 1-dimensional,
  the matrix-vector product is returned.
- If both arguments are at least 1-dimensional and at least one argument is
  N-dimensional (where N > 2), then a batched matrix multiply is returned.  If the first
  argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
  batched matrix multiply and removed after.  If the second argument is 1-dimensional, a
  1 is appended to its dimension for the purpose of the batched matrix multiple and removed after.
  The non-matrix (i.e. batch) dimensions are broadcasted (and thus
  must be broadcastable).  For example, if tensor1 is a (j x 1 x n x m) Tensor
  and tensor2 is a (k x m x p) Tensor, the returned tensor will be an (j x k x n x p) Tensor.
the original pytorch implementation is:
https://github.com/pytorch/pytorch/blob/v1.9.0/aten/src/ATen/native/LinearAlgebra.cpp#L1354
*/
nvinfer1::ITensor* MatmulConverter::converter(TensorrtEngine* engine,
                                            const torch::jit::Node *node,
                                            nvinfer1::ITensor* self,
                                            nvinfer1::ITensor* other) {
    auto self_dim = self->getDimensions().nbDims;
    auto other_dim = other->getDimensions().nbDims;
    auto origin_self_size = nvdim_to_sizes(self->getDimensions());
    auto origin_other_size = nvdim_to_sizes(other->getDimensions());

    LOG(INFO) << "self dim info : " << self->getDimensions()  << " and other dim  info: " << other->getDimensions();

    nvinfer1::ILayer* mm_layer = nullptr;
    //situation one: both tensors are 1D. this is like aten::dot
    if (self_dim == 1 && other_dim == 1) {
        mm_layer = engine->network()->addMatrixMultiply(
            *self, nvinfer1::MatrixOperation::kVECTOR, *other, nvinfer1::MatrixOperation::kVECTOR);

    //situation two: input tensor is 1D.
    } else if (self_dim == 1 && other_dim == 2) {
        mm_layer = engine->network()->addMatrixMultiply(
            *self, nvinfer1::MatrixOperation::kVECTOR, *other, nvinfer1::MatrixOperation::kNONE);

    //situation three: other tensor is 1D.
    } else if  (self_dim == 2 && other_dim == 1) {
        mm_layer = engine->network()->addMatrixMultiply(
            *self, nvinfer1::MatrixOperation::kNONE, *other, nvinfer1::MatrixOperation::kVECTOR);

    //situation four: input tensor is N-D(N > 2) and other tensor is 1D or 2D
    } else if (self_dim > 2 && (other_dim == 1 || other_dim == 2)) {
        if (other_dim == 1) {
            auto other_shuffle = engine->network()->addShuffle(*other);
            POROS_CHECK(other_shuffle, "Unable to create other shuffle layer for MatmulConverter");
            other_shuffle->setReshapeDimensions(unsqueeze_dims(other->getDimensions(), 1));
            other_shuffle->setName((layer_info(node) + "_IShuffleLayer_for_other").c_str());
            other = other_shuffle->getOutput(0);
            LOG(INFO) << "after shuffle other dim info turn to: " << other->getDimensions();
        }

        //prepare output_size info
        std::vector<int64_t> output_size;
        output_size.insert(output_size.end(), origin_self_size.begin(), origin_self_size.end() - 1);
        if (other_dim == 2) {
            auto other_size = nvdim_to_sizes(other->getDimensions());
            output_size.push_back(other_size[1]);
        }

        std::vector<int64_t> new_order = {-1, origin_self_size[self_dim -1]};
        auto self_shuffle = engine->network()->addShuffle(*self);
        POROS_CHECK(self_shuffle, "Unable to create self shuffle layer for MatmulConverter");
        self_shuffle->setReshapeDimensions(sizes_to_nvdim(new_order));
        self_shuffle->setName((layer_info(node) + "_IShuffleLayer_for_self").c_str());
        self = self_shuffle->getOutput(0);
        LOG(INFO) << "after shuffle self dim info turn to: " << self->getDimensions();

        auto tmp_mm_layer = engine->network()->addMatrixMultiply(
            *self, nvinfer1::MatrixOperation::kNONE, *other, nvinfer1::MatrixOperation::kNONE);
        POROS_CHECK(tmp_mm_layer, "Unable to create matrixmul layer for MatmulConverter");
        tmp_mm_layer->setName((layer_info(node) + "_IMatrixMultiplyLayer").c_str());
        auto tmp_output = tmp_mm_layer->getOutput(0);
        LOG(INFO) << "matmul output dim info : " << tmp_output->getDimensions();

        auto out_shuffle = engine->network()->addShuffle(*tmp_output);
        POROS_CHECK(out_shuffle, "Unable to create shuffle layer for MatmulConverter");
        out_shuffle->setReshapeDimensions(sizes_to_nvdim(output_size));
        self_shuffle->setName((layer_info(node) + "_IShuffleLayer_for_out").c_str());
        auto output = out_shuffle->getOutput(0);
        LOG(INFO) << "reshape output back to original dim info : " << tmp_output->getDimensions();
        return output;

    //situation five: input tensor is N-D(N > 2) and other tensor is 1D or 2D
    } else if (other_dim > 2 && (self_dim == 1 || self_dim == 2)) {
        const int64_t n = self_dim == 2 ? origin_self_size[0] : 1;
        const int64_t m = origin_self_size[self_dim - 1];
        const int64_t p = origin_other_size[other_dim - 1];

        //let's do other.transpose(-1, -2)
        std::vector<int64_t> new_order;
        for (int i = 0; i < other_dim; i++) {
            new_order.push_back(i);
        }
        new_order[other_dim - 1] = new_order[other_dim - 2];
        new_order[other_dim - 2] = other_dim - 1;
        auto other_shuffle = engine->network()->addShuffle(*other);
        POROS_CHECK(other_shuffle, "Unable to create shuffle layer from node: " << *node);
        nvinfer1::Permutation permute;
        std::copy(new_order.begin(), new_order.end(), permute.order);
        other_shuffle->setSecondTranspose(permute);
        other_shuffle->setName((layer_info(node) + "_IShuffleLayer_for_other").c_str());
        other = other_shuffle->getOutput(0);
        LOG(INFO) << "after transpose other dim info turn to: " << other->getDimensions();

        //self_T = self_dim == 2 ? self.t() : self.reshape({n, m}).t();
        if (self_dim == 1) {
            //tensor1.reshape({n, m})
            std::vector<int64_t> new_shape;
            new_shape = torch::reshape(torch::rand(origin_self_size), {n, m}).sizes().vec();
            auto tmp_shuffle = engine->network()->addShuffle(*self);
            POROS_CHECK(tmp_shuffle, "Unable to create shuffle layer for MatmulConverter");
            tmp_shuffle->setReshapeDimensions(sizes_to_nvdim(new_shape));
            tmp_shuffle->setName((layer_info(node) + "_IShuffleLayer_for_self_tmp").c_str());
            self = tmp_shuffle->getOutput(0);
            LOG(INFO) << "after reshape self dim info turn to: " << self->getDimensions();
        }
        //self.t()
        auto self_shuffle = engine->network()->addShuffle(*self);
        POROS_CHECK(self_shuffle, "Unable to create shuffle layer for MatmulConverter");
        nvinfer1::Permutation first_perm;
        first_perm.order[0] = 1;
        first_perm.order[1] = 0;
        self_shuffle->setFirstTranspose(first_perm);
        self_shuffle->setZeroIsPlaceholder(false);
        self_shuffle->setName((layer_info(node) + "_IShuffleLayer_for_self").c_str());
        self = self_shuffle->getOutput(0);
        LOG(INFO) << "after transpose self dim info turn to: " << self->getDimensions();

        //求 other.t() 与 self.t() 的matmul 的结果。
        auto mm_output = converter(engine, node, other, self);
        POROS_CHECK(mm_output, "Unable to calculate transpose matmul for MatmulConverter");
        auto mm_dim = mm_output->getDimensions().nbDims;
        auto mm_dim_size = nvdim_to_sizes(mm_output->getDimensions());

        //给我转置回来... 要哭了
        if (self_dim == 2) {
            std::vector<int64_t> new_order;
            for (int i = 0; i < mm_dim; i++) {
                new_order.push_back(i);
            }
            new_order[mm_dim - 1] = new_order[mm_dim - 2];
            new_order[mm_dim - 2] = mm_dim - 1;
            auto mm_shuffle = engine->network()->addShuffle(*mm_output);
            POROS_CHECK(mm_shuffle, "Unable to create shuffle layer for MatmulConverter");
            nvinfer1::Permutation permute;
            std::copy(new_order.begin(), new_order.end(), permute.order);
            mm_shuffle->setSecondTranspose(permute);
            mm_shuffle->setName((layer_info(node) + "_IShuffleLayer_for_output").c_str());
            auto output = mm_shuffle->getOutput(0);
            LOG(INFO) << "after transpose back ouput info turn to: " << output->getDimensions();
            return output;
        } else {
            //res_tensor.reshape(shape)
            std::vector<int64_t> shape;
            for (int i = 0; i < other_dim - 2; i++) {
                shape.push_back(origin_other_size[i]);
            }
            shape.push_back(p);
            auto new_shape = torch::reshape(torch::rand(mm_dim_size), shape).sizes().vec();
            auto mm_shuffle = engine->network()->addShuffle(*mm_output);
            POROS_CHECK(mm_shuffle, "Unable to create shuffle layer for MatmulConverter");
            mm_shuffle->setReshapeDimensions(sizes_to_nvdim(new_shape));
            mm_shuffle->setName((layer_info(node) + "_IShuffleLayer_for_output").c_str());
            auto output = mm_shuffle->getOutput(0);
            LOG(INFO) << "after transpose back ouput info turn to: " << output->getDimensions();
            return output;
        }

    } else {
        // expanding the dimensions if necessary.
        if (self->getDimensions().nbDims < other->getDimensions().nbDims) {
            auto newDims = self->getDimensions();
            for (int dim = self->getDimensions().nbDims; dim < other->getDimensions().nbDims; ++dim) {
                newDims = unsqueeze_dims(newDims, 0, 1, false);
            }
            LOG(INFO) << "Original self shape: " << self->getDimensions() << ", reshaping to: " << newDims;
            auto shuffle_layer = engine->network()->addShuffle(*self);
            POROS_CHECK(shuffle_layer, "Unable to create shuffle layer for MatmulConverter");
            shuffle_layer->setReshapeDimensions(newDims);
            shuffle_layer->setZeroIsPlaceholder(false);
            shuffle_layer->setName((layer_info(node) + "_IShuffleLayer_for_self").c_str());
            self = shuffle_layer->getOutput(0);
            //self = add_padding(engine, node, self, other->getDimensions().nbDims, false, false);
        } else if (other->getDimensions().nbDims < self->getDimensions().nbDims) {
            auto newDims = other->getDimensions();
            for (int dim = other->getDimensions().nbDims; dim < self->getDimensions().nbDims; ++dim) {
                newDims = unsqueeze_dims(newDims, 0, 1, false);
            }
            LOG(INFO) << "Original other shape: " << other->getDimensions() << ", reshaping to: " << newDims;
            auto shuffle_layer = engine->network()->addShuffle(*other);
            POROS_CHECK(shuffle_layer, "Unable to create shuffle layer for MatmulConverter");
            shuffle_layer->setReshapeDimensions(newDims);
            shuffle_layer->setZeroIsPlaceholder(false);
            shuffle_layer->setName((layer_info(node) + "_IShuffleLayer_for_other").c_str());
            other = shuffle_layer->getOutput(0);
            //other = add_padding(engine, node, other, self->getDimensions().nbDims, false, false);
        }

        mm_layer = engine->network()->addMatrixMultiply(
            *self, nvinfer1::MatrixOperation::kNONE, *other, nvinfer1::MatrixOperation::kNONE);
    }

    mm_layer->setName((layer_info(node) + "_IMatrixMultiplyLayer_for_other").c_str());
    POROS_CHECK(mm_layer, "Unable to create matrix multiplication node: " << *node);
    auto output = mm_layer->getOutput(0);
    return output;
}

bool MatmulConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for MatmulConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())),
        "input[0] for MatmulConverter is not Tensor as expected");
    POROS_CHECK_TRUE((inputs[1]->type()->isSubtypeOf(c10::TensorType::get())),
        "input[1] for MatmulConverter is not Tensor as expected");

    auto self = engine->context().get_tensor(inputs[0]);
    auto other = engine->context().get_tensor(inputs[1]);
    POROS_CHECK_TRUE(((self != nullptr) && (other != nullptr)),
        "Unable to init input tensor for node: " << *node);

    //add more log info for matmulConverter
    LOG(INFO) << "input[0] tensor is: " << node_info(inputs[0]->node());
    LOG(INFO) << "input[1] tensor is: " << node_info(inputs[1]->node());

    auto ouput = converter(engine, node, self, other);
    if (ouput != nullptr) {
        engine->context().set_tensor(node->outputs()[0], ouput);
        LOG(INFO) << "Output tensor shape: " << ouput->getDimensions();
        return true;
    } else {
        return false;
    }
}

/* aten::bmm(Tensor self, Tensor mat2) -> Tensor */
bool BmmConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for BmmConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for BmmConverter is not Tensor as expected");
    POROS_CHECK_TRUE((inputs[1]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[1] for BmmConverter is not Tensor as expected");


    auto self = engine->context().get_tensor(inputs[0]);
    auto mat2 = engine->context().get_tensor(inputs[1]);
    POROS_CHECK_TRUE(((self != nullptr) && (mat2 != nullptr)), 
        "Unable to init input tensor for node: " << *node);
    
    nvinfer1::Dims selfDims = self->getDimensions();
    nvinfer1::Dims mat2Dims = mat2->getDimensions();

    // check dimensions
    POROS_CHECK(selfDims.nbDims == 3,
        "Expected 3-dimensional tensor, but got " << selfDims.nbDims
        << "-dimensional tensor for argument #1 'batch1' (while checking arguments for bmm)");
    POROS_CHECK(mat2Dims.nbDims == 3,
        "Expected 3-dimensional tensor, but got " << mat2Dims.nbDims
        << "-dimensional tensor for argument #2 'batch2' (while checking arguments for bmm)");

    // Self and mat2 should have same size at dimension 0
    POROS_CHECK(selfDims.d[0] == mat2Dims.d[0],
        "Expected tensor to have size " << selfDims.d[0] << " at dimension 0, but got size " << mat2Dims.d[0]
        << " for argument #2 'batch2' (while checking arguments for bmm)");
    
    // The size of mat2 at dimension 1 should be the same as that of self at dimension 2.
    POROS_CHECK(selfDims.d[2] == mat2Dims.d[1],
        "Expected tensor to have size " << selfDims.d[2] << " at dimension 1, but got size " << mat2Dims.d[1]
        << " for argument #2 'batch2' (while checking arguments for bmm)");

    auto mm_layer = engine->network()->addMatrixMultiply(
        *self, nvinfer1::MatrixOperation::kNONE, *mat2, nvinfer1::MatrixOperation::kNONE);
    POROS_CHECK(mm_layer, "Unable to create matrix multiplication node: " << *node);
    
    mm_layer->setName((layer_info(node) + "_IMatrixMultiplyLayer").c_str());
    engine->context().set_tensor(node->outputs()[0], mm_layer->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << mm_layer->getOutput(0)->getDimensions();
    return true;
}

/** 
 * aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor 
 * check the function in pytorch: aten/src/ATen/RegisterSparseCuda.cpp  
 * at::native::addmm_sparse_dense_cuda(self, mat1, mat2, beta, alpha)
 * and the docs is like this: https://pytorch.org/docs/stable/generated/torch.addmm.html
 * 
 * %out: Tensor = aten::addmm(%bias, %mat1, %mat2, %beta, %alpha)
 * according to the torch.addmm explanation. the result is:
 * out = %beta * %bias + %alpha (%mat1 @ %mat2 )
 * 
 *  try to converter matmul like below:
 * %mm: Tensor = aten::matmul(%mat1, %mat2)
 * %bias_new: Tensor = aten::mul(%bias, %beta)
 * %out: Tensor = aten::add(%bias_new, %mm, %alpha) 
 **/
bool AddmmConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 5), "invaid inputs size for AddmmConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for AddmmConverter is not Tensor as expected");
    POROS_CHECK_TRUE((inputs[1]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[1] for AddmmConverter is not Tensor as expected");
    POROS_CHECK_TRUE((inputs[2]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[2] for AddmmConverter is not Tensor as expected");

    //extract bias & mat1 & mat2 
    auto bias = engine->context().get_tensor(inputs[0]);
    auto mat1 = engine->context().get_tensor(inputs[1]);
    auto mat2 = engine->context().get_tensor(inputs[2]);
    POROS_CHECK_TRUE(((bias != nullptr) && (mat1 != nullptr) && (mat2 != nullptr)), 
        "Unable to init input tensor for node: " << *node);

    //extract beta & alpha
    auto beta = (engine->context().get_constant(inputs[3])).toScalar().to<float>();
    auto alpha = (engine->context().get_constant(inputs[4])).toScalar().to<float>();

    /*-----------------------------------------------------------------------------
            step1: %mm: Tensor = aten::matmul(%mat1, %mat2)
    -------------------------------------------------------------------------------*/
    // Ensure mat1 and mat2 tensors have same nbDims by expanding the dimensions (from 0 axis) if
    // necessary.
    // TODO: this is too much simpler than the reality. we should change this someday
    if (mat1->getDimensions().nbDims < mat2->getDimensions().nbDims) {
        mat1 = add_padding(engine, node, mat1, mat2->getDimensions().nbDims, false, false);
    } else {
        mat2 = add_padding(engine, node, mat2, mat1->getDimensions().nbDims, false, false);
    }
    auto mm_layer = engine->network()->addMatrixMultiply(
        *mat1, nvinfer1::MatrixOperation::kNONE, *mat2, nvinfer1::MatrixOperation::kNONE);
    POROS_CHECK(mm_layer, "Unable to create matrix multiplication node: " << *node);
    mm_layer->setName((layer_info(node) + "_IMatrixMultiplyLayer").c_str());
    auto mm_output = mm_layer->getOutput(0);

    /*-----------------------------------------------------------------------------
            step2: %bias_new: Tensor = aten::mul(%bias, %beta)
    -------------------------------------------------------------------------------*/
    if (1 != beta) {
        auto beta_tensor = tensor_to_const(engine, torch::tensor({beta}));
        auto bias_new_layer = add_elementwise(engine, 
                            nvinfer1::ElementWiseOperation::kPROD,
                            bias,
                            beta_tensor,
                            layer_info(node) + "_prod_for_beta");
        POROS_CHECK(bias_new_layer, "Unable to create bias mul layer from node: " << *node);
        bias = bias_new_layer->getOutput(0);
    }
    
    /*-----------------------------------------------------------------------------
            step3: %out: Tensor = aten::add(%bias_new, %mm, %alpha)
    -------------------------------------------------------------------------------*/
    if (1 != alpha) {
        auto alpha_tensor = tensor_to_const(engine, torch::tensor({alpha}));
        auto mm_new_layer = add_elementwise(engine,
                            nvinfer1::ElementWiseOperation::kPROD,
                            mm_output,
                            alpha_tensor,
                            layer_info(node) + "_prod_for_alpha");
        POROS_CHECK(mm_new_layer, "Unable to create alpha*input layer from node: " << *node);
        mm_output = mm_new_layer->getOutput(0);
    }
    auto add_mm = add_elementwise(engine, 
            nvinfer1::ElementWiseOperation::kSUM, 
            bias,
            mm_output,
            layer_info(node) + "_sum");
    POROS_CHECK(add_mm, "Unable to create add layer from node: " << *node);
    engine->context().set_tensor(node->outputs()[0], add_mm->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << add_mm->getOutput(0)->getDimensions();
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, MatmulConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, BmmConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, AddmmConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
