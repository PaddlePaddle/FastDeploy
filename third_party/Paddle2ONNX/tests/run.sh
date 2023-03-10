# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
wget -P ~/.cache/paddle/dataset/int8/download/ http://paddle-inference-dist.bj.bcebos.com/int8/mnist_model.tar.gz
tar xf ~/.cache/paddle/dataset/int8/download/mnist_model.tar.gz -C ~/.cache/paddle/dataset/int8/download/
wget -P ~/.cache/paddle/dataset/int8/download/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar
tar xf ~/.cache/paddle/dataset/int8/download/MobileNetV1_infer.tar -C ~/.cache/paddle/dataset/int8/download/
wget -P ~/.cache/paddle/dataset/int8/download/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_infer.tar
tar xf ~/.cache/paddle/dataset/int8/download/ResNet50_infer.tar -C ~/.cache/paddle/dataset/int8/download/
wget -P ~/.cache/paddle/dataset/int8/download/ http://paddle-inference-dist.bj.bcebos.com/int8/calibration_test_data.tar.gz
mkdir ~/.cache/paddle/dataset/int8/download/small_data/ && tar xf ~/.cache/paddle/dataset/int8/download/calibration_test_data.tar.gz -C ~/.cache/paddle/dataset/int8/download/small_data/
wget https://bj.bcebos.com/paddle2onnx/tests/quantized_models.tar.gz
tar xf quantized_models.tar.gz

cases=`find . -name "test*.py" | sort`
ignore="test_auto_scan_affine_channel.py \
        test_auto_scan_expand.py \
        test_auto_scan_flatten2.py \
        test_auto_scan_generate_proposals_v2.py \
        test_auto_scan_matmul.py \
        test_auto_scan_mean.py \
        test_auto_scan_mul.py \
        test_auto_scan_multiclass_nms_lod.py \
        test_auto_scan_multiclass_nms_tensor.py \
        test_auto_scan_norm.py \
        test_auto_scan_roi_align.py \ # need to be rewrite
        test_expand_as.py \
        test_split.py \
        test_auto_scan_pool_adaptive_max_ops.py \
        test_auto_scan_interpolate_v1_ops.py \
        test_auto_scan_isx_ops.py \
        test_auto_scan_masked_select.py \
        test_auto_scan_pad2d.py \
        test_auto_scan_roll.py \
        test_auto_scan_set_value.py \
        test_auto_scan_top_k.py \
        test_auto_scan_unfold.py \
        test_auto_scan_uniform_random_batch_size_like.py \
        test_auto_scan_uniform_random.py \
        test_auto_scan_dist.py \
        test_auto_scan_distribute_fpn_proposals1.py \
        test_auto_scan_distribute_fpn_proposals_v2.py \
        test_auto_scan_generate_proposals.py \
        test_uniform.py \
        test_ceil.py \
        test_floor_divide.py \
        test_has_nan.py \
        test_isfinite.py \
        test_isinf.py \
        test_isnan.py \
        test_mask_select.py \
        test_median.py \
        test_nn_Conv3DTranspose.py \
        test_nn_GroupNorm.py \
        test_nn_InstanceNorm3D.py \
        test_nn_Upsample.py \
        test_normalize.py \
        test_scatter_nd_add.py \
        test_unsqueeze.py"
bug=0
export PY_CMD=$1
$PY_CMD -m pip install pytest
$PY_CMD -m pip uninstall onnxruntime
$PY_CMD -m pip install onnxruntime==1.11.1
export ENABLE_DEV=ON
echo "============ failed cases =============" >> result.txt
for file in ${cases}
do
    echo ${file}
    if [[ ${ignore} =~ ${file##*/} ]]; then
        echo "跳过"
    else
        $PY_CMD -m pytest ${file}
        if [ $? -ne 0 ]; then
            echo ${file} >> result.txt
            bug=`expr ${bug} + 1`
        fi
    fi
done

echo "total bugs: "${bug} >> result.txt
cat result.txt
exit ${bug}
