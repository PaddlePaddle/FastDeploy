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

#include "helper.h"
#include "cuda_multiprocess.h"

#if !defined(_WIN32)
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

// 可选：仅删除/解除共享内存命名对象（不依赖之前保存的 addr/fd）
static inline int sharedMemoryUnlinkByName(const char* name) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  // Windows 上没有 shm_unlink 语义。命名对象在最后一个句柄关闭后消失。
  // 这里做“尽力而为”：尝试打开后立即关闭，减少一次引用。
  HANDLE hMap = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, name);
  if (hMap) {
    CloseHandle(hMap);
    return 0;
  }
  // 已经不存在也算成功
  return 0;
#else
  // POSIX: 移除名字，未来不可再 open；已映射区仍存活直至 munmap
  if (shm_unlink(name) != 0) {
    if (errno == ENOENT) return 0;  // 不存在视作成功
    return errno;
  }
  return 0;
#endif
}

void UnsetDataIpc(const paddle::Tensor& tmp_input,
                         const std::string& shm_name,
                         bool close_ipc,
                         bool unlink_shm) {
  // 1) 关闭消费者导入的 IPC 映射（仅当 close_ipc=true 且该指针确为 OpenMemHandle 得来）
  if (close_ipc) {
    void* ptr = const_cast<void*>(tmp_input.data());
    checkCudaErrors(cudaIpcCloseMemHandle(ptr));
  }

  // 2) 解除共享内存命名对象（仅处理“名字”，不保证解除旧映射）
  if (unlink_shm) {
    int rc = sharedMemoryUnlinkByName(shm_name.c_str());
    if (rc != 0) {
      PD_THROW("Unlink shared memory failed: name=%s, err=%d",
               shm_name.c_str(), rc);
    }
  }
}

PD_BUILD_STATIC_OP(unset_data_ipc)
    .Inputs({"tmp_input"})
    .Attrs({"shm_name: std::string", "close_ipc: bool", "unlink_shm: bool"})
    .SetKernelFn(PD_KERNEL(UnsetDataIpc));
