# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import fastdeploy
from fastdeploy.text import UIEModel
import os
from pprint import pprint


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        required=True,
        help="The directory of model, params and vocab file.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        choices=['cpu', 'gpu'],
        help="Type of inference device, support 'cpu' or 'gpu'.")
    parser.add_argument(
        "--backend",
        type=str,
        default='onnx_runtime',
        choices=['onnx_runtime', 'paddle_inference', 'openvino'],
        help="The inference runtime backend.")
    parser.add_argument(
        "--cpu_num_threads",
        type=int,
        default=8,
        help="The number of threads to execute inference in cpu device.")
    return parser.parse_args()


def build_option(args):
    runtime_option = fastdeploy.RuntimeOption()
    # Set device
    if args.device == 'cpu':
        runtime_option.use_cpu()
    else:
        runtime_option.use_gpu()

    # Set backend
    if args.backend == 'onnx_runtime':
        runtime_option.use_ort_backend()
    elif args.backend == 'paddle_inference':
        runtime_option.use_paddle_backend()
    elif args.backend == 'openvino':
        runtime_option.use_openvino_backend()
    runtime_option.set_cpu_thread_num(args.cpu_num_threads)
    return runtime_option


if __name__ == "__main__":
    args = parse_arguments()
    runtime_option = build_option(args)

    model_path = os.path.join(args.model_dir, "inference.pdmodel")
    param_path = os.path.join(args.model_dir, "inference.pdiparams")
    vocab_path = os.path.join(args.model_dir, "vocab.txt")

    schema = ["时间", "选手", "赛事名称"]
    uie = UIEModel(
        model_path,
        param_path,
        vocab_path,
        position_prob=0.5,
        max_length=128,
        schema=schema,
        runtime_option=runtime_option)

    print("1. Named Entity Recognition Task")
    print(f"The extraction schema: {schema}")
    results = uie.predict(
        ["2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"], return_dict=True)
    pprint(results)
    print()

    schema = ["肿瘤的大小", "肿瘤的个数", "肝癌级别", "脉管内癌栓分级"]
    print(f"The extraction schema: {schema}")
    uie.set_schema(schema)
    results = uie.predict(
        [
            "（右肝肿瘤）肝细胞性肝癌（II-III级，梁索型和假腺管型），肿瘤包膜不完整，紧邻肝被膜，侵及周围肝组织，"
            "未见脉管内癌栓（MVI分级：M0级）及卫星子灶形成。（肿物1个，大小4.2×4.0×2.8cm）。"
        ],
        return_dict=True)
    pprint(results)
    print()

    print("2. Relation Extraction Task")
    schema = {"竞赛名称": ["主办方", "承办方", "已举办次数"]}
    print(f"The extraction schema: {schema}")
    uie.set_schema(schema)
    results = uie.predict(
        [
            "2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作"
            "委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。"
        ],
        return_dict=True)
    pprint(results)
    print()

    print("3. Event Extraction Task")
    schema = {"地震触发词": ["地震强度", "时间", "震中位置", "震源深度"]}
    print(f"The extraction schema: {schema}")
    uie.set_schema(schema)
    results = uie.predict(
        [
            "中国地震台网正式测定：5月16日06时08分在云南临沧市凤庆县(北纬24.34度，东经99.98度)发生3.5级地震，"
            "震源深度10千米。"
        ],
        return_dict=True)
    pprint(results)
    print()

    print("4. Opinion Extraction Task")
    schema = {"评价维度": ["观点词", "情感倾向[正向，负向]"]}
    print(f"The extraction schema: {schema}")
    uie.set_schema(schema)
    results = uie.predict(
        ["店面干净，很清静，服务员服务热情，性价比很高，发现收银台有排队"], return_dict=True)
    pprint(results)
    print()

    print("5. Sequence Classification Task")
    schema = ["情感倾向[正向，负向]"]
    print(f"The extraction schema: {schema}")
    uie.set_schema(schema)
    results = uie.predict(["这个产品用起来真的很流畅，我非常喜欢"], return_dict=True)
    pprint(results)
    print()

    print("6. Cross Task Extraction Task")
    schema = ["法院", {"原告": "委托代理人"}, {"被告": "委托代理人"}]
    print(f"The extraction schema: {schema}")
    uie.set_schema(schema)
    results = uie.predict(
        [
            "北京市海淀区人民法院\n民事判决书\n(199x)建初字第xxx号\n原告：张三。\n委托代理人李四，北京市 A律师"
            "事务所律师。\n被告：B公司，法定代表人王五，开发公司总经理。\n委托代理人赵六，北京市 C律师事务所律师。"
        ],
        return_dict=True)
    pprint(results)
