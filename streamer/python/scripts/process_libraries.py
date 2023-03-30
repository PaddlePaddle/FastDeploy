#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import shutil
import subprocess
import platform

user_specified_dirs = ['@OPENCV_DIRECTORY@', '@ORT_DIRECTORY@', ]

def process_on_linux(current_dir):
    rpaths = ["$ORIGIN:$ORIGIN/libs"]
    fd_libs = list()
    libs_path = os.path.join(current_dir, "streamer", "libs")
    for f in os.listdir(libs_path):
        filename = os.path.join(libs_path, f)
        if not os.path.isfile(filename):
            continue
        if f.count("fastdeploy") and f.count(".so") > 0:
            fd_libs.append(filename)

    cmake_build_dir = os.path.join(current_dir, ".setuptools-cmake-build")

    for lib in fd_libs:
        command = "{} --set-rpath '{}' {}".format(patchelf_bin_path, ":".join(rpaths), lib)
        if platform.machine() != 'sw_64' and platform.machine() != 'mips64':
            assert subprocess.Popen(
                command,
                shell=True) != 0, "patchelf {} failed, the command: {}".format(
                    command, lib)


def get_all_files(dirname):
    files = list()
    for root, dirs, filenames in os.walk(dirname):
        for f in filenames:
            fullname = os.path.join(root, f)
            files.append(fullname)
    return files


def process_libraries(current_dir):
    if platform.system().lower() == "linux":
        process_on_linux(current_dir)
    elif platform.system().lower() == "darwin":
        process_on_mac(current_dir)
    elif platform.system().lower() == "windows":
        process_on_windows(current_dir)

    all_files = get_all_files(os.path.join(current_dir, "streamer", "libs"))
    package_data = list()

    filters = [".vcxproj", ".png", ".java", ".h", ".cc", ".cpp", ".hpp"]
    for f in all_files:
        remain = True

        if remain:
            package_data.append(
                os.path.relpath(f, os.path.join(current_dir, "streamer")))
    return package_data
