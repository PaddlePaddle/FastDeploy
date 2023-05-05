# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table_model",
        required=True,
        help="Path of Table recognition model of PP-StructureV2.")
    parser.add_argument(
        "--table_char_dict_path",
        type=str,
        required=True,
        help="tabel recognition dict path.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'cpu' or 'gpu'.")
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Define which GPU card used to run model.")

    return parser.parse_args()


def build_option(args):

    table_option = fd.RuntimeOption()

    if args.device.lower() == "gpu":
        table_option.use_gpu(args.device_id)

    return table_option


args = parse_arguments()

table_model_file = os.path.join(args.table_model, "inference.pdmodel")
table_params_file = os.path.join(args.table_model, "inference.pdiparams")

# Set the runtime option
table_option = build_option(args)

# Create the table_model
table_model = fd.vision.ocr.StructureV2Table(
    table_model_file, table_params_file, args.table_char_dict_path,
    table_option)

# Read the image
im = cv2.imread(args.image)

# Predict and return the results
result = table_model.predict(im)

print(result)
