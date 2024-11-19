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

#include "fastdeploy/utils/utils.h"

#include <sstream>
#include <fstream>
#include <string_view>

#ifdef _WIN32
#include <Windows.h>
#endif

namespace fastdeploy {

bool FDLogger::enable_info = true;
bool FDLogger::enable_warning = true;

void SetLogger(bool enable_info, bool enable_warning) {
  FDLogger::enable_info = enable_info;
  FDLogger::enable_warning = enable_warning;
}

FDLogger::FDLogger(bool verbose, const std::string& prefix) {
  verbose_ = verbose;
  line_ = "";
#ifdef __ANDROID__
  prefix_ = std::string("[FastDeploy]") + prefix;
#else
  prefix_ = prefix;
#endif
}

FDLogger& FDLogger::operator<<(std::ostream& (*os)(std::ostream&)) {
  if (!verbose_) {
    return *this;
  }
  std::cout << prefix_ << " " << line_ << std::endl;
#ifdef __ANDROID__
  __android_log_print(ANDROID_LOG_INFO, prefix_.c_str(), "%s", line_.c_str());
#endif
  line_ = "";
  return *this;
}

// using os_string = std::filesystem::path::string_type;
#ifdef _WIN32
using os_string = std::wstring;
#else
using os_string = std::string;
#endif

os_string to_osstring(std::string_view utf8_str)
{
#ifdef _WIN32
    int len = MultiByteToWideChar(CP_UTF8, 0, utf8_str.data(), (int)utf8_str.size(), nullptr, 0);
    os_string result(len, 0);
    MultiByteToWideChar(CP_UTF8, 0, utf8_str.data(), (int)utf8_str.size(), result.data(), len);
    return result;
#else
    return std::string(utf8_str);
#endif
}

bool ReadBinaryFromFile(const std::string& path, std::string* contents)
{
  if (!contents) {
    return false;
  }
  auto& result = *contents;
  result.clear();

  std::ifstream file(to_osstring(path), std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return false;
  }

  auto fileSize = file.tellg();
  if (fileSize != -1) {
    result.resize(fileSize);
    file.seekg(0, std::ios::beg);
    file.read(const_cast<char*>(result.data()), fileSize);
  }
  else {
    // no size available, read to EOF
    constexpr auto chunksize = 4096;
    std::string chunk(chunksize, 0);
    while (!file.fail()) {
      file.read(const_cast<char*>(chunk.data()), chunksize);
      result.insert(result.end(), chunk.data(), chunk.data() + file.gcount());
    }
  }
  return true;
}

std::vector<int64_t> GetStride(const std::vector<int64_t>& dims) {
  auto dims_size = dims.size();
  std::vector<int64_t> result(dims_size, 1);
  for (int i = dims_size - 2; i >= 0; --i) {
    result[i] = result[i + 1] * dims[i + 1];
  }
  return result;
}

}  // namespace fastdeploy
