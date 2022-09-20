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


def process_on_linux(current_dir):
    rpaths = ["$ORIGIN:$ORIGIN/libs"]
    fd_libs = list()
    libs_path = os.path.join(current_dir, "fastdeploy", "libs")
    for f in os.listdir(libs_path):
        filename = os.path.join(libs_path, f)
        if not os.path.isfile(filename):
            continue
        if f.count("fastdeploy") and f.count(".so") > 0:
            fd_libs.append(filename)

    third_libs_path = os.path.join(libs_path, "third_libs")
    for root, dirs, files in os.walk(third_libs_path):
        for d in dirs:
            if d not in ["lib", "lib64"]:
                continue
            rel_path = os.path.relpath(os.path.join(root, d), libs_path)
            rpath = "$ORIGIN/" + rel_path
            rpaths.append(rpath)

    for lib in fd_libs:
        command = "patchelf --set-rpath '{}' {}".format(":".join(rpaths), lib)
        if platform.machine() != 'sw_64' and platform.machine() != 'mips64':
            assert subprocess.Popen(
                command,
                shell=True) != 0, "patchelf {} failed, the command: {}".format(
                    command, lib)


def process_on_mac(current_dir):
    fd_libs = list()
    libs_path = os.path.join(current_dir, "fastdeploy", "libs")
    for f in os.listdir(libs_path):
        filename = os.path.join(libs_path, f)
        if not os.path.isfile(filename):
            continue
        if f.count("fastdeploy") > 0 and (f.count(".dylib") > 0 or
                                          f.count(".so") > 0):
            fd_libs.append(filename)

    pre_commands = list()
    for lib in fd_libs:
        pre_commands.append(
            "install_name_tool -delete_rpath '@loader_path/libs' " + lib)

    commands = list()
    third_libs_path = os.path.join(libs_path, "third_libs")
    for root, dirs, files in os.walk(third_libs_path):
        for d in dirs:
            if d not in ["lib", "lib64"]:
                continue
        rel_path = rel_path = os.path.relpath(os.path.join(root, d), libs_path)
        rpath = "$loader_path/" + rel_path
        for lib in fd_libs:
            pre_commands.append(
                "install_name_tool -delete_rpath '@loader_path/{}' {}".format(
                    rpath, lib))
            commands.append("install_name_tool -add_rpath 'loader_path/{}' {}".
                            format(rpath, lib))

    for cmd in pre_commands:
        try:
            subprocess.Popen(cmd, shell=True)
        except:
            print("Skip execute command:", cmd)

    for cmd in commands:
        assert subprocess.Popen(
            cmd, shell=True) != 0, "Execute command failed: {}".format(cmd)


def process_on_windows(current_dir):
    libs_path = os.path.join(current_dir, "fastdeploy", "libs")
    third_libs_path = os.path.join(libs_path, "third_libs")
    for root, dirs, files in os.walk(third_libs_path):
        for f in files:
            file_path = os.path.join(root, f)
            if f.count('onnxruntime') > 0 and f.endswith('.dll'):
                shutil.copy(file_path, libs_path)


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

    all_files = get_all_files(os.path.join(current_dir, "fastdeploy", "libs"))
    package_data = list()

    if platform.system().lower() == "windows":

        def check_windows_legal_file(f):
            # Note(zhoushunjie): Special case for some library
            # File 'plugins.xml' is special case of openvino.
            for special_file in ['plugins.xml']:
                if special_file in f:
                    return True
            return False

        for f in all_files:
            if f.endswith(".pyd") or f.endswith("lib") or f.endswith(
                    "dll") or check_windows_legal_file(f):
                package_data.append(
                    os.path.relpath(f, os.path.join(current_dir,
                                                    "fastdeploy")))
        return package_data

    filters = [".vcxproj", ".png", ".java", ".h", ".cc", ".cpp", ".hpp"]
    for f in all_files:
        remain = True
        for flt in filters:
            if f.count(flt) > 0:
                remain = False
        filename = os.path.split(f)[-1]
        if filename in [
                "libnvinfer_plugin.so", "libnvinfer_plugin.so.8.4.1",
                "libnvinfer.so", "libnvinfer.so.8.4.1", "libnvonnxparser.so",
                "libnvonnxparser.so.8.4.1", "libnvparsers.so",
                "libnvparsers.so.8.4.1"
        ]:
            continue
        if remain:
            package_data.append(
                os.path.relpath(f, os.path.join(current_dir, "fastdeploy")))
    return package_data
