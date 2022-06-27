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

from __future__ import absolute_import
from six import text_type as _text_type
from .download import download, download_and_decompress

import argparse

# Since the source code is not fully open sourced,
# currently we will provide the prebuilt library
# and demo codes
import os

__version__ = "0.1.0"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=_text_type,
        default=None,
        help='Name of model, which can be listed by --list_models')
    parser.add_argument(
        '--platform',
        type=_text_type,
        default=None,
        help='Define platform, supports Windows/Linux/Android/iOS.')
    parser.add_argument(
        '--soc',
        type=_text_type,
        default=None,
        help='Define soc for the platform, supports x86/x86-NVIDIA_GPU/ARM/jetson.'
    )
    parser.add_argument(
        '--save_dir',
        type=_text_type,
        default=".",
        help='Path to download and extract deployment SDK.')
    parser.add_argument(
        '--list_models',
        required=False,
        action="store_true",
        default=False,
        help='List all the supported models.')
    parser.add_argument(
        '--download_sdk',
        required=False,
        action="store_true",
        default=False,
        help='List all the supported models.')

    return parser.parse_args()


def read_sources():
    user_dir = os.path.expanduser('~')
    print("Updating the newest sdk information...")
    source_cfgs = "https://bj.bcebos.com/paddlehub/fastdeploy/fastdeploy_newest_sources.cfg.1"
    if os.path.exists(os.path.join(user_dir, "fastdeploy_newest_sources.cfg.1")):
        os.remove(os.path.join(user_dir, "fastdeploy_newest_sources.cfg.1"))
    download(source_cfgs, user_dir)
    categories = dict()
    res = dict()
    with open(os.path.join(user_dir, "fastdeploy_newest_sources.cfg.1")) as f:
        for line in f:
            if line.strip().startswith("#"):
                continue
            if line.strip() == "":
                continue
            category, model, plat, soc, url = line.strip().split('\t')
            if category not in categories:
                categories[category] = set()
            categories[category].add(model)
            if model not in res:
                res[model] = dict()
            if plat not in res[model]:
                res[model][plat] = dict()
            if soc not in res[model][plat]:
                res[model][plat][soc] = dict()
            res[model][plat][soc] = url
    return categories, res


def main():
    args = parse_arguments()

    if not args.list_models and not args.download_sdk:
        print(
            "Please use flag --list_models to show all the supported models, or use flag --download_sdk to download the specify SDK to deploy you model."
        )
        return

    categories, all_sources = read_sources()
    all_models = list(all_sources.keys())
    all_models.sort()

    if args.list_models:
        print("Currently, FastDeploy supports {} models, list as below,\n".format(
            len(all_models)))

        for k, v in categories.items():
            print("\nModel Category: {}".format(k))
            print("_"*100)
            models = list(categories[k])
            models.sort()
            i = 0
            while i < len(models):
                if i == len(models) - 1:
                    print(models[i].center(30))
                    i += 1
                elif i == len(models) - 2:
                    print(models[i].center(30), models[i+1].center(30))
                    i += 2
                else:
                    print(models[i].center(30), models[i+1].center(30), models[i+2].center(30))
                    i += 3
        return

    if not os.path.exists(args.save_dir):
        print("The specified save_dir: {} is not exist.".format(args.save_dir))
        return

    if args.model is None or args.model == "":
        print(
            "Please define --model to choose which kind of model to deploy, use --list_models to show all the supported models."
        )
        return

    if args.model not in all_sources:
        print(
            "{} is not supported, use --list_models to list all the models FastDeploy supported.".
            format(args.model))
        return

    if args.platform is None or args.platform == "":
        print(
            "Please define --platform to choose which platform to deploy, supports windows/linux/android/ios."
        )
        return

    if args.platform not in all_sources[args.model]:
        print(
            "The model:{} only supports platform of {}, {} is not supported now.".
            format(args.model,
                   list(all_sources[args.model].keys()), args.platform))
        return

    if args.soc is None or args.soc == "":
        print(
            "Please define --soc to choose which hardware to deploy, for model:{} and platform:{}, the available socs are {}.".
            format(args.model, args.platform,
                   list(all_sources[args.model][args.platform].keys())))
        return

    if args.soc not in all_sources[args.model][args.platform]:
        print(
            "The model:{} in platform:{} only supports soc of {}, {} is not supported now.".
            format(args.model, args.platform,
                   list(all_sources[args.model][args.platform].keys()),
                   args.soc))
        return

    print("\nDownloading SDK:",
          all_sources[args.model][args.platform][args.soc])

    save_dir = args.save_dir
    sdk_name = os.path.split(all_sources[args.model][args.platform][args.soc])[
        -1].strip()
    if all_sources[args.model][args.platform][args.soc].count(".zip") > 0:
        sdk_name = os.path.split(all_sources[args.model][args.platform][
            args.soc])[-1].strip().split(".zip")[0]
        new_save_dir = os.path.join(args.save_dir, sdk_name)
        if not os.path.exists(new_save_dir):
            os.mkdir(new_save_dir)
        save_dir = new_save_dir
    download_and_decompress(
        all_sources[args.model][args.platform][args.soc],
        new_save_dir,
        rename=sdk_name + ".zip")
    os.remove(os.path.join(new_save_dir, sdk_name + ".zip"))
    print("Done. All the files of SDK have been extracted in {}.".format(
        new_save_dir))


if __name__ == "__main__":
    main()
