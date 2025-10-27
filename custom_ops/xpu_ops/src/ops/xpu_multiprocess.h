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

#pragma once

#include <errno.h>
#include <fcntl.h>
#include <memory.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
#include <xpu/runtime.h>
#include <xpu/version.h>
#include <vector>

struct shmStruct {
  size_t nprocesses;
#if XPURT_VERSION_MAJOR == 5
  XPUIpcMemHandle memHandle;
#endif
  uint64_t data_ptr_addr;
};

struct sharedMemoryInfo {
  void *addr;
  size_t size;
  int shmFd;
};

static int sharedMemoryCreate(const char *name,
                              size_t sz,
                              sharedMemoryInfo *info) {
  info->size = sz;

  info->shmFd = shm_open(name, O_RDWR | O_CREAT, 0777);
  PD_CHECK(info->shmFd >= 0, "shm_open failed");

  int status = ftruncate(info->shmFd, sz);
  PD_CHECK(status == 0, "ftruncate failed");

  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
  PD_CHECK(info->addr != NULL, "mmap failed");

  return 0;
}

static int sharedMemoryOpen(const char *name,
                            size_t sz,
                            sharedMemoryInfo *info) {
  info->size = sz;

  info->shmFd = shm_open(name, O_RDWR, 0777);
  PD_CHECK(info->shmFd >= 0, "shm_open failed");

  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
  PD_CHECK(info->addr != nullptr, "mmap failed");

  return 0;
}

static void sharedMemoryClose(sharedMemoryInfo *info) {
  if (info->addr) {
    munmap(info->addr, info->size);
  }
  if (info->shmFd) {
    close(info->shmFd);
  }
}
