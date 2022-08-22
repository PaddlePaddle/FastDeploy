// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef NON_64_PLATFORM

#include "fastdeploy/backends/ort/ops/distribute_fpn_proposals.h"
#include <algorithm>
#include <cmath>
#include "fastdeploy/core/fd_tensor.h"
#include "fastdeploy/utils/utils.h"
#include "fastdeploy/backends/ort/ops/utils.h"


namespace fastdeploy {

void DistributeFpnProposalsKernel::Compute(OrtKernelContext* context) {
  const OrtValue* fpn_rois = ort_.KernelContext_GetInput(context, 0);
  const OrtValue* rois_num = ort_.KernelContext_GetInput(context, 1);
  const float* fpn_rois_data =
      reinterpret_cast<const float*>(ort_.GetTensorData<float>(fpn_rois));
  const int32_t* rois_num_data =
      reinterpret_cast<const int32_t*>(ort_.GetTensorData<int32_t>(rois_num));
  OrtTensorDimensions fpn_rois_dim(ort_, fpn_rois);
  OrtTensorDimensions rois_num_dim(ort_, rois_num);

  FDASSERT(rois_num_dim.size() == 1, "Require dimension of input tensor:rois_num be 1, but now it's %d", rois_num_dim.size());

  int num_level = max_level - min_level + 1;

  std::vector<size_t> fpn_rois_lod(rois_num_dim[0]);
  fpn_rois_lod[0] = 0;
  for (size_t i = 0; i < rois_num_dim[0]; ++i) {
    fpn_rois_lod[i + 1] = fpn_rois_lod[i] + *(rois_num_data + i);
  }

  int fpn_rois_num = fpn_rois_lod[fpn_rois_lod.size() - 1];

  std::vector<int> target_level;
  std::vector<int> num_rois_level(num_level, 0);
  std::vector<int> num_rois_level_integral(num_level + 1, 0);
  for (size_t i = 0; i < fpn_rois_lod.size() - 1; ++i) {
    const float* rois_data = fpn_rois_data + fpn_rois_lod[i] * 4;
    for (int j = 0; j < fpn_rois_lod[i + 1]; ++j) {
      float roi_scale = std::sqrt(BBoxArea(rois_data, pixel_offset != 1));
      int64_t tgt_lvl = std::floor(std::log2(roi_scale / refer_scale + 1e-6) + refer_level);
      tgt_lvl = std::min(max_level, std::max(tgt_lvl, min_level));
      target_level.push_back(tgt_lvl);
      num_rois_level[tgt_lvl - min_level]++;
      rois_data += 4;
    }
  }

  std::vector<float*> multi_fpn_rois_data(num_level);
  std::vector<std::vector<size_t>> multi_fpn_rois_lod0;
  std::vector<OrtValue*> multi_fpn_rois(num_level, nullptr);
  std::vector<OrtValue*> multi_level_rois_num(num_level, nullptr);
  for (int i = 0; i < num_level; ++i) {
    std::vector<int64_t> out_dim = {num_rois_level[i], 4};
    multi_fpn_rois[i] = ort_.KernelContext_GetOutput(context, i, out_dim.data(), out_dim.size());
    multi_fpn_rois_data[i] = ort_.GetTensorMutableData<float>(multi_fpn_rois[i]);
    std::vector<size_t> lod0(1, 0);
    multi_fpn_rois_lod0.push_back(lod0);
    num_rois_level_integral[i + 1] = num_rois_level_integral[i] + num_rois_level[i];
  }

  std::vector<int64_t> index_dim = {fpn_rois_num, 1};
  OrtValue* restore_index = ort_.KernelContext_GetOutput(context, num_level * 2, index_dim.data(), index_dim.size());
  int32_t* restore_index_data = ort_.GetTensorMutableData<int32_t>(restore_index);

  std::vector<int> restore_index_inter(fpn_rois_num, -1);
  for (size_t i = 0; i < fpn_rois_lod.size() - 1; ++i) {
    const float* rois_data = fpn_rois_data + fpn_rois_lod[i] * 4;
    size_t cur_offset = fpn_rois_lod[i];
    for (int j = 0; j < num_level; ++j) {
      multi_fpn_rois_lod0[j].push_back(multi_fpn_rois_lod0[j][i]);
    }
    for (int j = 0; j < fpn_rois_lod[i + 1]; ++j) {
      int lvl = target_level[cur_offset + j];
      memcpy(multi_fpn_rois_data[lvl - min_level], rois_data, 4 * sizeof(float));
      multi_fpn_rois_data[lvl - min_level] += 4;
      int index_in_shuffle = num_rois_level_integral[lvl - min_level] + multi_fpn_rois_lod0[lvl - min_level][i + 1];
      restore_index_inter[index_in_shuffle] = cur_offset + j;
      multi_fpn_rois_lod0[lvl - min_level][i + 1]++;
      rois_data += 4;
    }
  }

  for (int i = 0; i < num_level; ++i) {
    restore_index_data[restore_index_inter[i]] = i;
  }
  
  for (int i = 0; i < num_level; ++i) {
    std::vector<int64_t> out_dim = {static_cast<int64_t>(fpn_rois_lod.size()) - 1};
    multi_level_rois_num[i] = ort_.KernelContext_GetOutput(context, i, out_dim.data(), out_dim.size());
    int32_t* multi_level_rois_num_data = ort_.GetTensorMutableData<int32_t>(multi_level_rois_num[i]);
    for (int j = 0; j < fpn_rois_lod.size() - 1; ++j) {
      multi_level_rois_num_data[j] = static_cast<int32_t>(multi_fpn_rois_lod0[i][j + 1] - multi_fpn_rois_lod0[i][j]);
    }
  }
}

void DistributeFpnProposalsKernel::GetAttribute(const OrtKernelInfo* info) {
  max_level = ort_.KernelInfoGetAttribute<int64_t>(info, "max_level");
  min_level = ort_.KernelInfoGetAttribute<int64_t>(info, "min_level");
  pixel_offset = ort_.KernelInfoGetAttribute<int64_t>(info, "pixel_offset");
  refer_level = ort_.KernelInfoGetAttribute<int64_t>(info, "refer_level");
  refer_scale = ort_.KernelInfoGetAttribute<int64_t>(info, "refer_scale");
}

}  // namespace fastdeploy

#endif
