// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <stdio.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/types.h>
#include "paddle/extension.h"

#define SPEC_LOGPROB_MAX_BSZ 512
#define SPEC_LOGPROB_K 20
#define MAX_DRAFT_TOKEN_NUM 6

struct batch_msgdata {
  int tokens[MAX_DRAFT_TOKEN_NUM * (SPEC_LOGPROB_K + 1)];
  float scores[MAX_DRAFT_TOKEN_NUM * (SPEC_LOGPROB_K + 1)];
  int ranks[MAX_DRAFT_TOKEN_NUM];
};

struct msgdata {
  long mtype;
  // stop_flag, message_flag, bsz, batch_token_nums
  int meta[3 + SPEC_LOGPROB_MAX_BSZ];
  batch_msgdata mtext[SPEC_LOGPROB_MAX_BSZ];
};
