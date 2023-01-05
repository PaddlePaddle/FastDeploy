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
import sys


def process_paddle_lite(paddle_lite_so_path):
    if platform.system().lower() != "linux":
        return
    rpaths = ["$ORIGIN", "$ORIGIN/mklml/lib/"]
    patchelf_exe = os.getenv("PATCHELF_EXE", "patchelf")

    
    for paddle_lite_so_file in os.listdir(paddle_lite_so_path):
        paddle_lite_so_file = os.path.join(paddle_lite_so_path,
                                           paddle_lite_so_file)

        # Patch /paddlelite/lib/*.so
        if '.so' in paddle_lite_so_file:
            command = "{} --set-rpath '{}' {}".format(
                patchelf_exe, ":".join(rpaths), paddle_lite_so_file)
            if platform.machine() != 'sw_64' and platform.machine(
            ) != 'mips64':
                assert os.system(
                    command) == 0, "patchelf {} failed, the command: {}".format(
                        paddle_lite_so_file, command)
        
         # Patch /paddlelite/lib/mklml/lib/*.so
        if 'mklml' in paddle_lite_so_file:
            paddle_lite_mklml_lib_path = os.path.join(paddle_lite_so_path,
                                           paddle_lite_so_file,'lib')
            
            for paddle_lite_mklml_so_file in os.listdir(paddle_lite_mklml_lib_path):
                paddle_lite_mklml_so_file = os.path.join(paddle_lite_mklml_lib_path,
                                           paddle_lite_mklml_so_file)

                if '.so' in paddle_lite_mklml_so_file:
                    command = "{} --set-rpath '{}' {}".format(
                        patchelf_exe, ":".join(rpaths), paddle_lite_mklml_so_file)
                    if platform.machine() != 'sw_64' and platform.machine(
                    ) != 'mips64':
                        assert os.system(
                            command) == 0, "patchelf {} failed, the command: {}".format(
                                paddle_lite_mklml_so_file, command) 
            

if __name__ == "__main__":
    process_paddle_lite(sys.argv[1])
