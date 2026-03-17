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
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

// Custom ftok that uses the low 20 bits of id instead of only 8 bits.
// This avoids dependency on filesystem paths while preserving queue separation.
inline key_t custom_ftok(const char* path, int id) {
  struct stat st;
  if (stat(path, &st) < 0) {
    fprintf(stderr,
            "[custom_ftok] stat(\"%s\") failed (errno=%d), "
            "msg queue key will be invalid!\n",
            path,
            errno);
    return static_cast<key_t>(-1);
  }
  // low 4 bits of st_dev | low 8 bits of st_ino | low 20 bits of id
  return static_cast<key_t>(((st.st_dev & 0x0f) << 28) |
                            ((st.st_ino & 0xff) << 20) | (id & 0xfffff));
}
