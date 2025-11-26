// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include <fcntl.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/msg.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdint>
#include <vector>
#include "msg_utils.h"  // NOLINT

struct RemoteCacheKvIpc {
  struct save_cache_kv_complete_signal_layerwise_meta_data {
    int32_t layer_id = -1;
    void* shm_ptr = nullptr;
    int shm_fd = -1;
    save_cache_kv_complete_signal_layerwise_meta_data() {}
    save_cache_kv_complete_signal_layerwise_meta_data(int32_t layer_id_,
                                                      void* shm_ptr_,
                                                      int shm_fd_)
        : layer_id(layer_id_), shm_ptr(shm_ptr_), shm_fd(shm_fd_) {}
  };

  struct save_cache_kv_complete_signal_layerwise_meta_data_per_query {
    int layer_id_;
    int num_layers_;
    bool inited = false;
    struct msgdatakv msg_sed;
    int msgid;

    save_cache_kv_complete_signal_layerwise_meta_data_per_query() {}

    void init(const int* seq_lens_encoder,
              const int* seq_lens_decoder,
              const int rank,
              const int num_layers,
              const int real_bsz) {
      layer_id_ = 0;
      num_layers_ = num_layers;
      msg_sed.mtype = 1;
      int encoder_count = 0;
      for (int i = 0; i < real_bsz; i++) {
        if (seq_lens_encoder[i] > 0) {
          msg_sed.mtext[3 * encoder_count + 2] = i;
          msg_sed.mtext[3 * encoder_count + 3] = seq_lens_decoder[i];
          msg_sed.mtext[3 * encoder_count + 4] = seq_lens_encoder[i];
          encoder_count++;
        }
      }
      msg_sed.mtext[0] = encoder_count;

      if (!inited) {
        // just init once
        const int msg_id = 1024 + rank;
        key_t key = ftok("/opt/", msg_id);
        msgid = msgget(key, IPC_CREAT | 0666);
        inited = true;
      }
    }

    void send_signal() {
      if (inited) {
        msg_sed.mtext[1] = layer_id_;
        if ((msgsnd(msgid, &msg_sed, (MAX_BSZ * 3 + 2) * 4, 0)) == -1) {
          printf("kv signal full msg buffer\n");
        }
        layer_id_ = (layer_id_ + 1);
        assert(layer_id_ <= num_layers_);
      }
    }
  };

  static RemoteCacheKvIpc::save_cache_kv_complete_signal_layerwise_meta_data
      kv_complete_signal_meta_data;
  static RemoteCacheKvIpc::
      save_cache_kv_complete_signal_layerwise_meta_data_per_query
          kv_complete_signal_meta_data_per_query;
  static void* kv_complete_signal_identity_ptr;
  static bool kv_complete_signal_shmem_opened;

  static RemoteCacheKvIpc::save_cache_kv_complete_signal_layerwise_meta_data
  open_shm_and_get_complete_signal_meta_data(const int rank_id,
                                             const int device_id,
                                             const bool keep_pd_step_flag);
  static void save_cache_kv_complete_signal_layerwise(void* meta_data);
  static void save_cache_kv_complete_signal_layerwise_per_query(
      void* meta_data);
};
