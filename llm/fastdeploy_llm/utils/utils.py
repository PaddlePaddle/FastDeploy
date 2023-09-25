# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import glob
import shutil
import os
import json
import struct
import numpy as np


def deserialize_from_file(fp):
    """
    从磁盘读取ids
    """
    # dtype
    x_type = fp.read(1)
    x_type_out = struct.unpack("c", x_type)[0]
    # data
    data_list = []
    if x_type_out == b'0':
        data = fp.read(4)
        while data:
            data_out = struct.unpack("f", data)[0]
            data_list.append(data_out)
            data = fp.read(4)
    elif x_type_out == b'1':
        data = fp.read(8)
        while data:
            data_out = struct.unpack("l", data)[0]
            data_list.append(data_out)
            data = fp.read(8)
    elif x_type_out == b'2':
        data = fp.read(4)
        while data:
            data_out = struct.unpack("i", data)[0]
            data_list.append(data_out)
            data = fp.read(4)
    else:
        print("type error", flush=True)
    data_arr = np.array(data_list).reshape(-1)
    return data_arr


def get_files(curr_dir, ext):
    for i in glob.glob(os.path.join(curr_dir, ext)):
        yield i


def remove_files(dir, ext):
    try:
        for i in get_files(dir, ext):
            os.remove(i)
    except Exception as e:
        logger.error("remove files error: {0}".format(str(e)))


def check_model(dir):
    is_static = False
    rank = 0
    if os.path.exists(os.path.join(dir, "model.pdmodel")) and os.path.exists(
            os.path.join(dir, "model.pdiparams")):
        is_static = True
        rank = 1
    else:
        for i in range(8):
            if os.path.exists(os.path.join(dir, "rank_{}".format(i))):
                if os.path.exists(
                        os.path.join(dir, "rank_{}".format(i),
                                     "model.pdmodel")) and os.path.exists(
                                         os.path.join(dir, "rank_{}".format(i),
                                                      "model.pdiparams")):
                    is_static = True
                    rank = i + 1
    return is_static, rank
