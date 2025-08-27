// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdio.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/types.h>
#include "paddle/extension.h"

#define MAX_BSZ 128
#define K 5

struct msgdata {
  long mtype;                        // NOLINT
  int mtext[MAX_BSZ * (K + 1) + 2];  // stop_flag, bsz, tokens
  float mtext_f[MAX_BSZ * (K + 1)];  // score
};

void SaveOutMmsgTopK(const paddle::Tensor& x,
                     const paddle::Tensor& scores,
                     const paddle::Tensor& topk_ids,
                     const paddle::Tensor& topk_scores,  // [bsz, k
                     const paddle::Tensor& not_need_stop,
                     int k,
                     int64_t rank_id) {
  if (rank_id > 0) return;
  auto x_cpu = x.copy_to(paddle::CPUPlace(), false);
  auto scores_cpu = scores.copy_to(paddle::CPUPlace(), false);
  auto topk_ids_cpu = topk_ids.copy_to(paddle::CPUPlace(), false);
  auto topk_scores_cpu = topk_scores.copy_to(paddle::CPUPlace(), false);
  int64_t* x_data = x_cpu.data<int64_t>();
  float* scores_data = scores_cpu.data<float>();
  int64_t* topk_ids_data = topk_ids_cpu.data<int64_t>();
  float* topk_scores_data = topk_scores_cpu.data<float>();
  static struct msgdata msg_sed;
  static key_t key = ftok("./", 1);
  static int msgid = msgget(key, IPC_CREAT | 0666);

  msg_sed.mtype = 1;
  bool not_need_stop_data = not_need_stop.data<bool>()[0];
  msg_sed.mtext[0] = not_need_stop_data ? 1 : -1;
  int bsz = x.shape()[0];
  msg_sed.mtext[1] = bsz;
  for (int i = 0; i < bsz; i++) {
    for (int j = 0; j < K + 1; j++) {
      const int offset = i * (K + 1) + j;
      if (j == 0) {
        msg_sed.mtext[offset + 2] = static_cast<int>(x_data[i]);
        msg_sed.mtext_f[offset] = scores_data[i];
      } else if (j <= k + 1) {
        msg_sed.mtext[offset + 2] =
            static_cast<int>(topk_ids_data[i * k + j - 1]);
        msg_sed.mtext_f[offset] = topk_scores_data[i * k + j - 1];
      } else {
        msg_sed.mtext[offset + 2] = -1;
        msg_sed.mtext_f[offset] = 0.0;
      }
    }
  }
  if ((msgsnd(msgid,
              &msg_sed,
              (MAX_BSZ * (K + 1) + 2) * 4 + (MAX_BSZ * (K + 1)) * 4,
              IPC_NOWAIT)) == -1) {
    printf("full msg buffer\n");
  }
  return;
}

PD_BUILD_OP(save_output_topk)
    .Inputs({"x", "scores", "topk_ids", "topk_scores", "not_need_stop"})
    .Attrs({"k: int", "rank_id: int64_t"})
    .Outputs({"x_out"})
    .SetInplaceMap({{"x", "x_out"}})
    .SetKernelFn(PD_KERNEL(SaveOutMmsgTopK));
