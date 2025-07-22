// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "remote_cache_kv_ipc.h"

RemoteCacheKvIpc::save_cache_kv_complete_signal_layerwise_meta_data RemoteCacheKvIpc::kv_complete_signal_meta_data;
RemoteCacheKvIpc::save_cache_kv_complete_signal_layerwise_meta_data_per_query
    RemoteCacheKvIpc::kv_complete_signal_meta_data_per_query;
void* RemoteCacheKvIpc::kv_complete_signal_identity_ptr = nullptr;
bool RemoteCacheKvIpc::kv_complete_signal_shmem_opened = false;

RemoteCacheKvIpc::save_cache_kv_complete_signal_layerwise_meta_data
                            RemoteCacheKvIpc::open_shm_and_get_complete_signal_meta_data(
                                                            const int rank_id,
                                                            const int device_id,
                                                            const bool keep_pd_step_flag) {
    if (RemoteCacheKvIpc::kv_complete_signal_shmem_opened){
        if (keep_pd_step_flag) {
            return RemoteCacheKvIpc::kv_complete_signal_meta_data;
        }
        RemoteCacheKvIpc::kv_complete_signal_meta_data.layer_id = -1;
        int32_t* layer_complete_ptr = reinterpret_cast<int32_t*>(kv_complete_signal_meta_data.shm_ptr);
        *layer_complete_ptr = -1;
        int32_t current_identity = (*reinterpret_cast<int32_t*>(RemoteCacheKvIpc::kv_complete_signal_identity_ptr));
        int32_t* write_ptr = reinterpret_cast<int32_t*>(RemoteCacheKvIpc::kv_complete_signal_identity_ptr);
        *write_ptr = (current_identity + 1) % 100003;
        return RemoteCacheKvIpc::kv_complete_signal_meta_data;
    }

    std::string flags_server_uuid;
    if (const char* iflags_server_uuid_env_p = std::getenv("SHM_UUID")){
        std::string iflags_server_uuid_env_str(iflags_server_uuid_env_p);
        flags_server_uuid = iflags_server_uuid_env_str;
    }

    std::string step_shm_name = ("splitwise_complete_prefilled_step_"
                    + std::to_string(rank_id) + "." + std::to_string(device_id));
    std::string layer_shm_name = ("splitwise_complete_prefilled_layer_"
                    + std::to_string(rank_id) + "." + std::to_string(device_id));
    if (const char* use_ep = std::getenv("ENABLE_EP_DP")){
        if(std::strcmp(use_ep, "1") == 0){
        step_shm_name = "splitwise_complete_prefilled_step_tprank0_dprank"
                + std::to_string(rank_id) + "_" + flags_server_uuid;
        layer_shm_name = "splitwise_complete_prefilled_layer_tprank0_dprank"
                + std::to_string(rank_id) + "_" + flags_server_uuid;
        }
    }

    int signal_shm_fd = shm_open(layer_shm_name.c_str(), O_CREAT | O_RDWR, 0666);

    PADDLE_ENFORCE_NE(signal_shm_fd,
                        -1,
                        phi::errors::InvalidArgument(
                            "can not open shm for cache_kv_complete_signal."));
    int signal_shm_ftruncate = ftruncate(signal_shm_fd, 4);
    void* signal_ptr = mmap(0, 4, PROT_WRITE, MAP_SHARED, signal_shm_fd, 0);

    PADDLE_ENFORCE_NE(
        signal_ptr,
        MAP_FAILED,
        phi::errors::InvalidArgument(
                            "MAP_FAILED for cache_kv_complete_signal."));
    int32_t* write_signal_ptr = reinterpret_cast<int32_t*>(signal_ptr);
    *write_signal_ptr = -1;
    using type_meta_data = RemoteCacheKvIpc::save_cache_kv_complete_signal_layerwise_meta_data;

    // std::printf("#### open_shm_and_get_complete_signal_meta_data layer idx:%d, to ptx:%p \n",
    //             -1, signal_ptr);

    type_meta_data meta_data(
        -1,
        signal_ptr,
        signal_shm_fd
    );
    RemoteCacheKvIpc::kv_complete_signal_meta_data = meta_data;
    int identity_shm_fd = shm_open(step_shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    PADDLE_ENFORCE_NE(identity_shm_fd,
                        -1,
                        phi::errors::InvalidArgument(
                            "can not open shm for cache_kv_complete_identity."));

    int identity_shm_ftruncate = ftruncate(identity_shm_fd, 4);
    void* identity_ptr = mmap(0, 4, PROT_WRITE, MAP_SHARED, identity_shm_fd, 0);
    PADDLE_ENFORCE_NE(
        identity_ptr,
        MAP_FAILED,
        phi::errors::InvalidArgument(
                            "MAP_FAILED for prefill_identity."));
    int32_t current_identity = (*reinterpret_cast<int32_t*>(identity_ptr));
    int32_t* write_ptr = reinterpret_cast<int32_t*>(identity_ptr);
    *write_ptr = (current_identity + 1) % 100003;
    RemoteCacheKvIpc::kv_complete_signal_identity_ptr = identity_ptr;
    RemoteCacheKvIpc::kv_complete_signal_shmem_opened = true;
    return meta_data;
}

void CUDART_CB RemoteCacheKvIpc::save_cache_kv_complete_signal_layerwise(void* meta_data) {
    int64_t* meta_data_ptr = reinterpret_cast<int64_t*>(meta_data);
    int32_t layer_id = meta_data_ptr[0];
    int32_t* ptr = reinterpret_cast<int32_t*>(meta_data_ptr[1]);
    *ptr = layer_id;
    // std::printf("#### save_cache_kv_complete_signal_layerwise layer idx:%d, to ptx:%p \n",
    //             *ptr, meta_data_ptr[1]);
}

void CUDART_CB RemoteCacheKvIpc::save_cache_kv_complete_signal_layerwise_per_query(void* meta_data) {
    RemoteCacheKvIpc::kv_complete_signal_meta_data_per_query.send_signal();
    // std::printf("#### save_cache_kv_complete_signal_layerwise_per_query);
}
