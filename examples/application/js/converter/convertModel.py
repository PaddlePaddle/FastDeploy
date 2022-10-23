#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import json
import collections
import math
import sys
import os
import struct
import argparse
import shutil
import stat
import traceback
import numpy as np
import paddle.fluid as fluid
import paddle as paddle
import copy
from functools import reduce
import rnn
from pruningModel import pruningNoSenseTensor
from fuseOps import opListFuse


# 输入模型所在目录
modelDir = None
# 输入模型名
modelName = None
# 输入参数名，当且仅当所有模型参数被保存在一个单独的二进制文件中，它才需要被指定，若为分片模型，请设置为None
paramsName = None
# 是否打印模型信息
enableLogModelInfo = False
# 输出模型目录
outputDir = None
# 分片文件大小，单位：KB
sliceDataSize = 4 * 1024
# paddlepaddle运行程序实例
program = None
# 存放模型结构
modelInfo = {"vars": {}, "ops": [], "chunkNum": 0, "dataLayout": "nchw", "feedShape": None}
# 存放参数数值（未排序）
paramValuesDict = {}

# 有一些后置算子适合在cpu中运行，所以单独统计
postOps = []
# 在转换过程中新生成的、需要添加到vars中的variable
appendedVarList = []
# rnn op索引列表
rnnList = []

# 转换模型中需要过滤掉的参数
needFilterAttributes = ['op_callstack', 'col', 'op_role', 'op_namescope', 'op_role_var',
    'data_format', 'is_test', 'use_mkldnn', 'use_cudnn', 'use_quantizer', 'workspace_size_MB',
    'mkldnn_data_type', 'op_device', '__@kernel_type_attr@__']


class ObjDict(dict):
    """
    Makes a  dictionary behave like an object,with attribute-style access.
    """
    def __getattr__(self,name):
        try:
            return self[name]
        except:
            raise AttributeError(name)
    def __setattr__(self,name,value):
        self[name]=value

def validateShape(shape, name):
    """检验shape长度，超过4则截断"""
    if len(shape) > 4:
        newShape = shape[-4:]
        print('\033[31m ' + name + ' tensor shape length > 4, 处理为丢弃头部shape \033[0m')
        return newShape
    return shape

def splitLargeNum(x):
   """将x拆分成两个因数相乘"""
   # 获取最小值
   num = math.floor(math.sqrt(x))
   while (num):
        if x % num == 0:
           return [num, int(x / num)]
        num -= 1

   return [1, x]

def logModel(info):
    """ 打印信息 """
    if enableLogModelInfo:
        print(info)

def sortDict(oldDict, reverse=False):
    """ 对字典进行排序，返回有序字典，默认升序 """
    # 获得排序后的key list
    keys = sorted(oldDict.keys(), reverse=reverse)
    orderDict = collections.OrderedDict()
    # 遍历 key 列表
    for key in keys:
        orderDict[key] = oldDict[key]
    return orderDict

def dumpModelToJsonFile(outputDir):
    """ 导出模型数据到json文件 """
    print("Dumping model structure to json file...")
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    outputModelPath = os.path.join(outputDir, "model.json")
    with open(outputModelPath, 'w') as outputFile:
        json.dump(modelInfo, outputFile, indent=4, separators=(", ", ": "), sort_keys=True)
    print("Dumping model structure to json file successfully")

def sliceDataToBinaryFile(paramValueList, outputDir):
    """ 将参数数据分片输出到文件，默认分片策略为按4M分片 """
    totalParamValuesCount = len(paramValueList)
    countPerSlice = int(sliceDataSize * 1024 / 4)

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    currentChunkIndex = 0
    currentParamDataIndex = 0

    while currentParamDataIndex < totalParamValuesCount - 1:
        remainCount = totalParamValuesCount - currentParamDataIndex
        if remainCount < countPerSlice:
            countPerSlice = remainCount
        chunkPath = os.path.join(outputDir, 'chunk_%s.dat' % (currentChunkIndex + 1))
        file = open(chunkPath, 'wb')
        for i in paramValueList[currentParamDataIndex : currentParamDataIndex + countPerSlice]:
            byte = struct.pack('f', float(i))
            file.write(byte)
        file.close()
        currentParamDataIndex = currentParamDataIndex + countPerSlice
        currentChunkIndex = currentChunkIndex + 1
        print("Output No." + str(currentChunkIndex)+ " binary file, remain " + str(totalParamValuesCount - currentParamDataIndex) + " param values.")
    print("Slicing data to binary files successfully. (" + str(currentChunkIndex)+ " output files and " + str(currentParamDataIndex) + " param values)")

def reorderParamsValue():
    """ 对参数文件中的数值，按照variable.name字母序排序，返回排序后组合完成的value list """
    paramValuesOrderDict = sortDict(paramValuesDict)
    paramValues = []
    for value in paramValuesOrderDict.values():
        paramValues += value
    return paramValues


def mapToPaddleJSTypeName(fluidOPName):
    """ 处理fluid的OP type与PaddleJS的OP type不对应情况 """
    if fluidOPName == "batch_norm":
        return "batchnorm"
    return fluidOPName

def excludeNegativeShape(shape):
    varShape = list(shape)
    varShapeExcludeNegativeOne = []
    for s in varShape:
        # 模型中 ？会自动转为 -1，需要单独处理成 1
        if s == -1:
            s = 1
        varShapeExcludeNegativeOne.append(s)
    return varShapeExcludeNegativeOne

def organizeModelVariableInfo(result):
    """ 组织参数信息 """
    print("Organizing model variables info...")
    index = 0
    # 存放var信息（未排序）
    varInfoDict = {}
    # 获取program中所有的var，遍历并获取所有未排序的var信息和参数数值
    vars = list(program.list_vars())
    for v in vars:
        # 跳过feed和fetch
        if "feed" == v.name:
            continue
        if "fetch" == v.name:
            continue

        varShape = excludeNegativeShape(v.shape)
        # FIXME:end

        # 存放variable信息，在dump成json时排序
        varInfo = {}
        varInfo["shape"] = varShape
        # 数据是否是持久化数据，如tensor为持久化数据，op的output不是持久化数据
        # 只输出持久化数据，paddlejs中也仅读取持久化数据
        varInfo["persistable"] = v.persistable
        varInfoDict[v.name] = varInfo

        logModel("[Var index:" + str(index) + " name:" + v.name + "]")
        jsonDumpsIndentStr = json.dumps(varInfo, indent=2)
        logModel(jsonDumpsIndentStr)
        logModel("")
        index += 1

        # persistable数据存入paramValuesDict，等待排序
        if v.persistable:
            tensor = np.array(fluid.global_scope().find_var(v.name).get_tensor())
            data = tensor.flatten().tolist()
            paramValuesDict[v.name] = data

    # shape推断校正
    feed_target_names = result[1]
    fetch_targets = result[2]
    # 获取输入shape
    feedData = {}
    feeded_vars = [program.global_block().vars[varname] for varname in feed_target_names]
    for feedItem in feeded_vars:
        curShape = feedItem.shape
        feedName = feedItem.name
        feedData[feedName] = np.full(excludeNegativeShape(curShape), 1.0, "float32")

    for v in program.list_vars():
        if not v.persistable:
            v.persistable = True
    exe.run(program, feed=feedData, fetch_list=fetch_targets, return_numpy=False)

    for varKey in varInfoDict:
        var = fluid.global_scope().find_var(varKey)
        varData = np.array(var.get_tensor())
        varShape = list(varData.shape)
        varInfoDict[varKey]['shape'] = validateShape(varShape, varKey)

    # vars追加
    vars = modelInfo['vars']
    for appendedVar in appendedVarList:
        appendedName = appendedVar['name']
        newName = appendedVar['new']
        for curVarKey in varInfoDict:
            if curVarKey == appendedName:
                newVar = copy.deepcopy(varInfoDict[curVarKey])
                varInfoDict[newName] = newVar
                break
    # 对var信息dict，按照key（var名）进行字母顺序排序
    varInfoOrderDict = sortDict(varInfoDict)
    # 将var信息按照顺序，添加到model info的vars中
    for key, value in varInfoOrderDict.items():
        value["name"] = key
        modelInfo["vars"][key] = value
    print("Organizing model variables info successfully.")

def organizeModelOpInfo():
    """ 组织模型OP结构信息 """
    print("Organizing model operators info...")
    ops = program.current_block().ops
    feedOutputName = None
    index = 0
    for op in ops:
        opInfo = {}

        # 获取OP type，需要映射到PaddleJS的名字
        opInfo["type"] = mapToPaddleJSTypeName(op.type)

        opInputs = op.input_names
        opOutputs = op.output_names

        # 获取OP input
        inputs = {}
        for name in opInputs:
            value = op.input(name)
            if len(value) <= 0:
                continue
            if value[0] == feedOutputName:
                # FIXME:workaround,PaddleJSfeed 输入必须是image，且为单输入，这里修改feed后面的OP的input为image，建立前后关联
                inputs[name] = ["image"]
            else:
                inputs[name] = value
        opInfo["inputs"] = inputs

        # 获取OP output
        outputs = {}
        # 将outputs转换为数组
        if op.type == 'density_prior_box' or op.type == 'prior_box' or op.type == 'box_coder':
            outputs['Out'] = []
            for name in opOutputs:
                value = op.output(name)
                if len(value) <= 0:
                    continue
                outputs['Out'].append(value[0])
        else:
            for name in opOutputs:
                value = op.output(name)
                if len(value) <= 0:
                    continue
                if op.type == "feed":
                    # FIXME:workaround,PaddleJSfeed 输入必须是image，且为单输入，这里保存原始的输出名，以便映射
                    feedOutputName = value[0]
                    outputs[name] = ["image"]
                else:
                    outputs[name] = value

        opInfo["outputs"] = outputs

        # 收敛outputs[name]
        if "Output" in opInfo["outputs"]:
            opInfo["outputs"]["Out"] = opInfo["outputs"]["Output"]
            del opInfo["outputs"]["Output"]

        elif "Y" in opInfo["outputs"]:
            opInfo["outputs"]["Out"] = opInfo["outputs"]["Y"]
            del opInfo["outputs"]["Y"]

        if "Out" not in opInfo["outputs"]:
            print("\033[31moutputs[name] not exist Out.\033[0m")
            sys.exit(1)

        # 有的模型如人脸关键点，会出现两个算子合并的情况，如lmk_demo，elementwise_add后接了relu算子，relu的输入输出相等，兼容一下
        # inputs与outputs只有一个，名称相等，则，输入加后缀，改上一层算子。
        if 'X' in inputs and 'Out' in outputs:
            curInputs = inputs['X']
            curOutputs = outputs['Out']
            if len(curInputs) == 1 and len(curOutputs) == 1 and curInputs[0] == curOutputs[0] and index > 1:
                originName = curInputs[0]
                changedName = inputs['X'][0] = curInputs[0] = originName + '_changed'
                opInfo["inputs"]['X'] = curInputs
                # 获取上一层算子
                prevOpOutputs = modelInfo["ops"][index - 1]['outputs']
                for name in prevOpOutputs:
                    values = prevOpOutputs[name]
                    for i, curName in enumerate(values):
                        if (curName == originName):
                            modelInfo["ops"][index - 1]['outputs'][name][i] = changedName
                appendedVarList.append({'name': originName, 'new': changedName})

        # 获取OP attribute
        attrs = {}
        for name in op.attr_names:
            # 过滤不需要的参数
            if name in needFilterAttributes:
                continue
            value = op.attr(name)
            attrs[name] = value
        opInfo["attrs"] = attrs

        if (op.type == 'rnn'):
            global rnnList
            rnnList.append(index)

        # multiclass_nms 单独处理
        if (op.type.startswith('multiclass_nms')):
            opInfo["type"] = 'multiclass_nms'
            postOps.append(opInfo)
        else:
            # 存入modelInfo
            modelInfo["ops"].append(opInfo)
        logModel("[OP index:" + str(index) + " type:" + op.type + "]")
        jsonDumpsIndentStr = json.dumps(opInfo, indent=2)
        logModel(jsonDumpsIndentStr)
        logModel("")
        index += 1
    print("Organizing model operators info successfully.")

def addChunkNumToJson(paramValueList):
    totalParamValuesCount = len(paramValueList)
    countPerSlice = int(sliceDataSize * 1024 / 4)
    count = totalParamValuesCount / countPerSlice
    modelInfo["chunkNum"] = math.ceil(count)
    print("Model chunkNum set successfully.")

def appendConnectOp(fetch_targets):
    targets = []
    inputNames = []
    totalShape = 0

    # 从fetch_targets中提取输出算子信息
    for target in fetch_targets:
        name = target.name
        curVar = fluid.global_scope().find_var(name)
        curTensor = np.array(curVar.get_tensor())
        shape = list(curTensor.shape)
        totalShape += reduce(lambda x, y: x * y, shape)
        targets.append({'name': name, 'shape': excludeNegativeShape(shape)})
        inputNames.append(name)

    # 构造connect算子
    op = {
        'attrs': {},
        'inputs': {'X': inputNames},
        'outputs': {'Out': ['connect_result']},
        'type': 'connect'
    }
    # 构造输出var
    outputVar = {'name': 'connect_result', 'shape': splitLargeNum(totalShape)}

    ops = modelInfo['ops']
    vars = modelInfo['vars']

    # 收集要删除的算子index
    delList = []
    for index, item in enumerate(ops):
        if item['type'] == 'fetch':
            delList.append(index)

    # 去除fetch算子
    delCount = 0
    for delIndex in delList:
        del ops[delIndex - delCount]
        delCount += 1

    fetchOp = {
        "attrs": {},
        "inputs": {
            "X": [
                "connect_result"
            ]
        },
        "outputs": {
            "Out": [
                "fetch"
            ]
        },
        "type": "fetch"
    }
    ops.append(op)
    ops.append(fetchOp)

    vars['connect_result'] = outputVar
    modelInfo['multiOutputs'] = targets
    return targets

def genModelFeedShape(feed):
    if len(feed) != 1:
        print("\033[33;1mModel has more than one input feed.\033[0m")
        return

    originFeedShape = modelInfo['vars'][feed[0]]['shape']
    feedShape = {}
    if len(originFeedShape) == 3:
        feedShape['fc'] = originFeedShape[0]
        feedShape['fh'] = originFeedShape[1]
        feedShape['fw'] = originFeedShape[2]
    elif len(originFeedShape) == 4:
        feedShape['fc'] = originFeedShape[1]
        feedShape['fh'] = originFeedShape[2]
        feedShape['fw'] = originFeedShape[3]
    elif len(originFeedShape) == 2:
        feedShape['fh'] = originFeedShape[0]
        feedShape['fw'] = originFeedShape[1]
    else:
        print("\033[33;1mFeedShape length is " + str(len(originFeedShape)) + ".\033[0m")
        return

    modelInfo['feedShape'] = feedShape
    print("\033[32mModel FeedShape set successfully.\033[0m")

def convertToPaddleJSModel(modelDir, modelName, paramsName, outputDir, useGPUOpt):
    """ 转换fluid modle为paddleJS model """


    #In PaddlePaddle 2.x, we turn on dynamic graph mode by default, and 'load_inference_model()' is only supported in static graph mode. So  call 'paddle.enable_static()' before this api to enter static graph mode.
    paddle.enable_static()

    # 初始化fluid运行环境和配置
    global exe
    exe = fluid.Executor(fluid.CPUPlace())
    result = fluid.io.load_inference_model(dirname=modelDir, executor=exe, model_filename=modelName, params_filename=paramsName)
    global program
    program = result[0]
    fetch_targets = result[2]
    feed_target_names = result[1]

    # 获取program中所有的op，按op顺序加入到model info
    organizeModelOpInfo()

    # 获取program中所有的var，按照字母顺序加入到model info，同时读取参数数值
    organizeModelVariableInfo(result)

    # 拆分rnn op
    if len(rnnList):
        for index in rnnList:
            rnn.splice_rnn_op(modelInfo, index)

    if useGPUOpt:
        # 算子融合
        modelInfo['gpuOpt'] = True
        opListFuse(modelInfo['ops'])

    # 对多输出模型追加connect算子
    if len(fetch_targets) > 1:
        appendConnectOp(fetch_targets)

    if (postOps and len(postOps) > 0):
        for op in postOps:
            if (op['type'].startswith('multiclass_nms')):
                inputNames = []
                for input, value in op['inputs'].items():
                    if len(value) <= 0:
                        continue
                    cur = ObjDict()
                    cur.name = value[0]
                    inputNames.append(cur)
                targets = appendConnectOp(inputNames)
                # op['inputs'] = targets
                keys = op['inputs'].keys()
                for i, val in enumerate(keys):
                    op['inputs'][val] = targets[i]


        modelInfo['postOps'] = postOps

    # 对参数数值dict，按照key（参数名）进行字母顺序排序，并组合到一起
    paramValues = reorderParamsValue()

    # model.json 设置分片参数
    addChunkNumToJson(paramValues)

    # model.json 设置 feedShape 输入 shape 信息
    genModelFeedShape(feed_target_names)

    # 去掉无意义的 tensor 和对应 op
    pruningNoSenseTensor(modelInfo)

    # 导出模型文件到json
    dumpModelToJsonFile(outputDir)

    # 导出分片参数文件
    sliceDataToBinaryFile(paramValues, outputDir)

def main():

    global sliceDataSize
    global enableLogModelInfo

    try:
        p = argparse.ArgumentParser(description='模型转换参数解析')
        p.add_argument('--inputDir', help='fluid模型所在目录。当且仅当使用分片参数文件时使用该参数。将过滤modelPath和paramsPath参数，且模型文件名必须为`__model__`', required=False)
        p.add_argument('--modelPath', help='fluid模型文件所在路径，使用合并参数文件时使用该参数', required=False)
        p.add_argument('--paramPath', help='fluid参数文件所在路径，使用合并参数文件时使用该参数', required=False)
        p.add_argument("--outputDir", help='paddleJS模型输出路径，必要参数', required=True)
        p.add_argument("--logModelInfo", type=int, default=0, help='是否输出模型结构信息，非必要参数，0为不输出，1为输出，默认不输出', required=False)
        p.add_argument("--sliceDataSize", type=int, default=4096, help='分片输出参数文件时，每片文件的大小，单位：KB，非必要参数，默认4096KB', required=False)
        p.add_argument('--useGPUOpt', help='转换模型是否执行GPU优化方法', required=False)

        args = p.parse_args()
        modelDir = args.inputDir
        modelPath = args.modelPath
        paramPath = args.paramPath
        useGPUOpt = args.useGPUOpt

        if not modelDir:
            modelDir, modelName = os.path.split(modelPath)
            paramDir, paramsName = os.path.split(paramPath)
            if paramDir != modelDir:
                print("\033[31mModel and param file should be put in a same directory!\033[0m")
                raise Exception()
        outputDir = args.outputDir
        sliceDataSize = args.sliceDataSize

        if args.logModelInfo == 1:
            enableLogModelInfo = True

        convertToPaddleJSModel(modelDir, modelName, paramsName, outputDir, useGPUOpt)

    except Exception as identifier:
        print("\033[31mA fetal error occured. Failed to convert model.\033[0m")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
