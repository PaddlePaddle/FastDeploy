#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import collections
import argparse
import traceback
from paddlejslite import lite
import pkg_resources
from packaging import version

lite_version = pkg_resources.get_distribution("paddlelite").version

def optimizeModel(inputDir, modelPath, paramPath, outputDir):
    """ 使用opt python接口执行模型优化 """
    opt = lite.Opt()
    if inputDir:
        # 分片参数文件优化
        opt.set_model_dir(inputDir)
    else:
        # 合并参数文件优化
        opt.set_model_file(modelPath)
        opt.set_param_file(paramPath)

    opt.set_valid_places("arm")
    opt.set_model_type("protobuf")
    opt.set_optimize_out(outputDir)
    opt.run()


def main():
    try:
        p = argparse.ArgumentParser('模型优化参数解析')
        p.add_argument('--inputDir', help='fluid模型所在目录。当且仅当使用分片参数文件时使用该参数。将过滤modelPath和paramsPath参数，且模型文件名必须为`__model__`', required=False)
        p.add_argument('--modelPath', help='fluid模型文件所在路径，使用合并参数文件时使用该参数', required=False)
        p.add_argument('--paramPath', help='fluid参数文件所在路径，使用合并参数文件时使用该参数', required=False)
        p.add_argument("--outputDir", help='优化后fluid模型目录，必要参数', required=True)

        args = p.parse_args()
        inputDir = args.inputDir
        modelPath = args.modelPath
        paramPath = args.paramPath
        outputDir = args.outputDir

        optimizeModel(inputDir, modelPath, paramPath, outputDir)

    except Exception as identifier:
        print("\033[31mA fetal error occured. Failed to optimize model.\033[0m")
        print(traceback.format_exc())
        pass


if __name__ == "__main__":
    main()
