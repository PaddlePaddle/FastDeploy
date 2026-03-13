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

#include <cassert>
#include <cstdio>
#include <set>
#include <string>
#include <sys/ipc.h>
#include <sys/stat.h>
#include <sys/types.h>

// Inline the function under test directly to avoid paddle dependencies.
inline key_t custom_ftok(const char* path, int id) {
  struct stat st;
  if (stat(path, &st) < 0) {
    return static_cast<key_t>(-1);
  }
  return static_cast<key_t>(((st.st_dev & 0xff) << 24) |
                            ((st.st_ino & 0xff) << 16) | (id & 0xffff));
}

// Test 1: ids in [0, 65536) must produce unique keys for the same path.
void test_unique_keys_for_same_path() {
  const char* path = "/dev/shm";
  std::set<key_t> keys;
  int collisions = 0;
  for (int id = 0; id < 65536; id++) {
    key_t k = custom_ftok(path, id);
    assert(k != static_cast<key_t>(-1));
    if (!keys.insert(k).second) {
      collisions++;
      fprintf(
          stderr, "  COLLISION: id=%d produced duplicate key=0x%x\n", id, k);
    }
  }
  assert(collisions == 0);
  assert(keys.size() == 65536);
  printf(
      "[PASS] test_unique_keys_for_same_path: 65536 ids -> 65536 unique "
      "keys\n");
}

// Test 2: different paths must produce different keys for the same id.
void test_different_paths_different_keys() {
  const char* paths[] = {"/opt/", "/dev/shm", "/tmp"};
  int num_paths = sizeof(paths) / sizeof(paths[0]);

  // Verify all paths are accessible first.
  for (int i = 0; i < num_paths; i++) {
    struct stat st;
    if (stat(paths[i], &st) < 0) {
      printf("[SKIP] test_different_paths_different_keys: %s not accessible\n",
             paths[i]);
      return;
    }
  }

  int test_ids[] = {0, 1, 255, 256, 1024, 32768, 65535};
  int num_ids = sizeof(test_ids) / sizeof(test_ids[0]);

  for (int t = 0; t < num_ids; t++) {
    int id = test_ids[t];
    std::set<key_t> keys;
    for (int i = 0; i < num_paths; i++) {
      key_t k = custom_ftok(paths[i], id);
      assert(k != static_cast<key_t>(-1));
      bool inserted = keys.insert(k).second;
      if (!inserted) {
        fprintf(stderr,
                "  COLLISION: path=%s id=%d produced duplicate key=0x%x\n",
                paths[i],
                id,
                k);
      }
      assert(inserted);
    }
  }
  printf(
      "[PASS] test_different_paths_different_keys: %d paths x %d ids, all "
      "unique\n",
      num_paths,
      num_ids);
}

// Test 3: standard ftok only uses low 8 bits, so id=1 and id=257 collide.
//         custom_ftok must NOT collide for these.
void test_no_collision_beyond_8bits() {
  const char* path = "/dev/shm";
  // Standard ftok would map 1 and 257 to the same key (both & 0xff == 1).
  key_t k1 = custom_ftok(path, 1);
  key_t k257 = custom_ftok(path, 257);
  assert(k1 != k257);

  key_t k0 = custom_ftok(path, 0);
  key_t k256 = custom_ftok(path, 256);
  assert(k0 != k256);

  key_t k255 = custom_ftok(path, 255);
  key_t k511 = custom_ftok(path, 511);
  assert(k255 != k511);

  printf(
      "[PASS] test_no_collision_beyond_8bits: id pairs (1,257), (0,256), "
      "(255,511) all differ\n");
}

// Test 4: invalid path should return -1.
void test_invalid_path() {
  key_t k = custom_ftok("/nonexistent_path_xyz_12345", 42);
  assert(k == static_cast<key_t>(-1));
  printf("[PASS] test_invalid_path: returns -1 for nonexistent path\n");
}

int main() {
  printf("=== custom_ftok unit tests ===\n");
  test_unique_keys_for_same_path();
  test_different_paths_different_keys();
  test_no_collision_beyond_8bits();
  test_invalid_path();
  printf("=== ALL TESTS PASSED ===\n");
  return 0;
}
