#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import os
import argparse
import shutil
import stat
import traceback
import copy


def cleanTempModel(optimizedModelTempDir):
    """ 清理opt优化完的临时模型文件 """
    if os.path.exists(optimizedModelTempDir):
        print("Cleaning optimized temporary model...")
        shutil.rmtree(optimizedModelTempDir, onerror=grantWritePermission)


def grantWritePermission(func, path, execinfo):
    """ 文件授权 """
    os.chmod(path, stat.S_IWRITE)
    func(path)


def main():
    """
    Example:
    'python convertToPaddleJSModel.py --modelPath=../infer_model/MobileNetV2/model --paramPath=../infer_model/MobileNetV2/params --outputDir=../jsmodel'
    """
    try:
        p = argparse.ArgumentParser(description='转化为PaddleJS模型参数解析')
        p.add_argument(
            '--inputDir',
            help='fluid模型所在目录。当且仅当使用分片参数文件时使用该参数。将过滤modelPath和paramsPath参数，且模型文件名必须为`__model__`',
            required=False)
        p.add_argument(
            '--modelPath', help='fluid模型文件所在路径，使用合并参数文件时使用该参数', required=False)
        p.add_argument(
            '--paramPath', help='fluid参数文件所在路径，使用合并参数文件时使用该参数', required=False)
        p.add_argument(
            "--outputDir", help='paddleJS模型输出路径，必要参数', required=True)
        p.add_argument(
            "--disableOptimize",
            type=int,
            default=0,
            help='是否关闭模型优化，非必要参数，1为关闭优化，0为开启优化，默认开启优化',
            required=False)
        p.add_argument(
            "--logModelInfo",
            type=int,
            default=0,
            help='是否输出模型结构信息，非必要参数，0为不输出，1为输出，默认不输出',
            required=False)
        p.add_argument(
            "--sliceDataSize",
            type=int,
            default=4096,
            help='分片输出参数文件时，每片文件的大小，单位：KB，非必要参数，默认4096KB',
            required=False)
        p.add_argument('--useGPUOpt', help='转换模型是否执行GPU优化方法', required=False)

        args = p.parse_args()
        # 获取当前用户使用的 python 解释器 bin 位置
        pythonCmd = sys.executable

        # TODO: 由于Paddle Lite和PaddlePaddle存在包冲突，因此将整个模型转换工具拆成两个python文件，由一个入口python文件通过命令行调用
        # 区分本地执行和命令行执行
        if os.path.exists("optimizeModel.py"):
            optimizeCmd = pythonCmd + " optimizeModel.py"
        else:
            optimizeCmd = "pdjsOptimizeModel"

        if os.path.exists("convertModel.py"):
            convertCmd = pythonCmd + " convertModel.py"
        else:
            convertCmd = "pdjsConvertModel"

        inputDir = args.inputDir
        modelPath = args.modelPath
        paramPath = args.paramPath
        outputDir = args.outputDir
        disableOptimization = args.disableOptimize
        args.disableOptimize = None
        enableLogModelInfo = args.logModelInfo
        sliceDataSize = args.sliceDataSize

        optArgs = copy.deepcopy(args)

        enableOptimization = 1 - disableOptimization
        optimizedModelTempDir = None
        if enableOptimization == 1:
            optimizedModelTempDir = os.path.join(outputDir, "optimize")
            optArgs.outputDir = optimizedModelTempDir
            if inputDir:
                args.inputDir = optimizedModelTempDir
            else:
                args.modelPath = os.path.join(optimizedModelTempDir, "model")
                args.paramPath = os.path.join(optimizedModelTempDir, "params")

        print("============Convert Model Args=============")
        if inputDir:
            print("inputDir: " + inputDir)
        else:
            print("modelPath: " + modelPath)
            print("paramPath: " + paramPath)
        print("outputDir: " + outputDir)
        print("enableOptimizeModel: " + str(enableOptimization))
        print("enableLogModelInfo: " + str(enableLogModelInfo))
        print("sliceDataSize:" + str(sliceDataSize))

        print("Starting...")
        if enableOptimization:
            print("Optimizing model...")
            for param in ["inputDir", "modelPath", "paramPath", "outputDir"]:
                if optArgs.__dict__[param]:
                    # 用""框起命令参数值，解决路径中的空格问题
                    optimizeCmd += " --" + param + "=" + '"' + str(
                        optArgs.__dict__[param]) + '"'
            os.system(optimizeCmd)
            try:
                os.listdir(optimizedModelTempDir)
            except Exception as identifier:
                print("\n\033[31mOptimizing model failed.\033[0m")
                # restore inputDir or modelPath paramPath from optimize
                if inputDir:
                    args.inputDir = inputDir
                else:
                    args.modelPath = modelPath
                    args.paramPath = paramPath
            else:
                print("\n\033[32mOptimizing model successfully.\033[0m")
        else:
            print(
                "\033[33mYou choosed not to optimize model, consequently, optimizing model is skiped.\033[0m"
            )

        print("\nConverting model...")
        for param in args.__dict__:
            if args.__dict__[param]:
                # 用""框起参数，解决路径中的空格问题
                convertCmd += " --" + param + "=" + '"' + str(args.__dict__[
                    param]) + '"'
        os.system(convertCmd)
        try:
            file = os.listdir(outputDir)
            if len(file) == 0:
                raise Exception
        except Exception as identifier:
            print("\033[31m============ALL DONE============\033[0m")
        else:
            if enableOptimization:
                cleanTempModel(optimizedModelTempDir)
                print("Temporary files has been deleted successfully.")
            print("\033[32mConverting model successfully.\033[0m")
            print("\033[32m============ALL DONE============\033[0m")

    except Exception as identifier:
        print("\033[31mA fetal error occured. Failed to convert model.\033[0m")
        print(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()
