#!/usr/bin/env python
# -*- coding: UTF-8 -*-

def opListFuse(ops):
    """ 算子融合 """
    fuseOpList = [
        'relu',
        'relu6',
        'leaky_relu',
        'scale',
        'sigmoid',
        'hard_sigmoid',
        'pow',
        'sqrt',
        'tanh',
        'hard_swish',
        'dropout'
    ]

    # 判断op是否为单节点
    def opExistSingleNode(opName):
        name = opName
        if name:
            nodeNum = 0
            for i in range(len(ops)):
                op = ops[i]
                if 'X' not in op['inputs']:
                    continue

                inputName = op['inputs']['X']
                for x in inputName:
                    if x == name:
                        nodeNum = nodeNum + 1

            return True if nodeNum == 1 else False

        else:
            return False


    for index in reversed(range(len(ops))):
        if index > 0:
            op = ops[index]

            # 兼容paddlelite 算子融合字段
            if 'act_type' in op['attrs']:
                name = op['attrs']['act_type']
                op['attrs']['fuse_opt'] = {}
                op['attrs']['fuse_opt'][name] = {}

                if name == 'hard_swish':
                    op['attrs']['fuse_opt'][name]['offset'] = op['attrs']['hard_swish_offset']
                    op['attrs']['fuse_opt'][name]['scale'] = op['attrs']['hard_swish_scale']
                    op['attrs']['fuse_opt'][name]['threshold'] = op['attrs']['hard_swish_threshold']

                if name == 'relu6':
                    op['attrs']['fuse_opt'][name]['threshold'] = op['attrs']['fuse_brelu_threshold']

            for fuse in fuseOpList:
                if op['type'] == fuse:
                    prevOp = ops[index - 1]

                    if opExistSingleNode(prevOp['outputs']['Out'][0]) and len(prevOp['outputs']['Out']) == 1 :
                        prevOp['attrs']['fuse_opt'] = {}
                        if 'fuse_opt' in op['attrs']:
                            prevOp['attrs']['fuse_opt'] = op['attrs']['fuse_opt']
                            del op['attrs']['fuse_opt']

                        prevOp['attrs']['fuse_opt'][fuse] = op['attrs']
                        prevOp['outputs']['Out'] = op['outputs']['Out']

                        del ops[index]



