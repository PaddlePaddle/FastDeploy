#!/usr/bin/env python
# -*- coding: UTF-8 -*-

def splice_rnn_op(model_info, rnn_index):

    global input_shape
    global weight_0_shape
    global weight_1_shape
    global rnn_input_name
    ops = model_info['ops']
    vars = model_info['vars']
    op = ops[rnn_index]

    rnn_input_name = op['inputs']['Input'][0]
    rnn_output_name = op['outputs']['Out'][0]

    is_bidirec = 2 if op['attrs']['is_bidirec'] else 1
    num_layers = op['attrs']['num_layers']
    hidden_size = op['attrs']['hidden_size']
    layer_num = num_layers * is_bidirec
    # concat input最大值
    max_concat_num = 15

    def concat_mul(index, list, num):
        global rnn_input_name
        end = len(list)

        if end < max_concat_num:
            concat_output_name = 'lstm_' + str(index - 1) + '_' + str(num) + '.tmp_concat'
            # 非最后一层遍历，将concat作为下一层输入
            if index < is_bidirec * num_layers - 1:
                rnn_input_name = concat_output_name

            # 最后一层遍历，将rnn_output_name赋给最后一个concat
            else:
                concat_output_name = rnn_output_name

            concat_op = {
                'attrs': {
                    'axis': 0
                },
                'inputs': {
                    'X': []
                },
                'outputs': {'Out': [concat_output_name]},
                'type': 'concat'
            }

            concat_output_shape = 0

            for x in range(0, end):
                x_input_name = 'lstm_' + str(index - 1) + '_' + str(list[x]) + '.tmp_concat'
                concat_op['inputs']['X'].append(x_input_name)
                concat_output_shape += vars[x_input_name]['shape'][0]

            concat_var = {
                'name': concat_output_name,
                'persistable': False,
                'shape': [concat_output_shape, 1, weight_1_shape[1] * 2]
            }

            ops.append(concat_op)

            if index < is_bidirec * num_layers - 1:
                vars[concat_output_name] = concat_var
            return

        # concat新列表
        new_list = []

        for i in range(0, end, max_concat_num):
            if i + max_concat_num > end:
                for n in range(i, end):
                    new_list.append(list[n])
                break

            concat_output_name = 'lstm_' + str(index - 1) + '_' + str(num) + '.tmp_concat'
            # concat_list长度为max_concat_num && 最后一层遍历，将rnn_output_name赋给最后一个concat
            if end == max_concat_num and index == is_bidirec * num_layers - 1:
                concat_output_name = rnn_output_name

            concat_op = {
                'attrs': {
                    'axis': 0
                },
                'inputs': {
                    'X': []
                },
                'outputs': {'Out': [concat_output_name]},
                'type': 'concat'
            }

            concat_output_shape = 0

            for x in range(0, max_concat_num):
                x_input_name = 'lstm_' + str(index - 1) + '_' + str(list[i + x]) + '.tmp_concat'
                concat_op['inputs']['X'].append(x_input_name)
                concat_output_shape += vars[x_input_name]['shape'][0]

            concat_var = {
                'name': concat_output_name,
                'persistable': False,
                'shape': [concat_output_shape, 1, weight_1_shape[1] * 2]
            }
        
            ops.append(concat_op)
            vars[concat_output_name] = concat_var
            new_list.append(num)

            # 若concat_list长度为max_concat_num，在下一次递归时直接修改rnn_input_name，结束递归，num无需+1
            if end != max_concat_num:
                num += 1

        concat_mul(index, new_list, num)

    for index in range(layer_num):
        last_hidden = op['inputs']['PreState'][0]
        last_cell = op['inputs']['PreState'][1]
        weight_list_0 = op['inputs']['WeightList'][index * 2]
        weight_list_1 = op['inputs']['WeightList'][index * 2 + 1]
        weight_list_2 = op['inputs']['WeightList'][(index + num_layers * is_bidirec) * 2]
        weight_list_3 = op['inputs']['WeightList'][(index + num_layers * is_bidirec) * 2 + 1]
        output_name = 'rnn_origin_' + str(index)
        input_shape = vars[rnn_input_name]['shape']
        batch = input_shape[0]

        if vars[weight_list_0]:
            weight_0_shape = vars[weight_list_0]['shape']

        if vars[weight_list_1]:
            weight_1_shape = vars[weight_list_1]['shape']

        if batch == 0:
            continue

        origin_op = {
            'attrs': {
                'state_axis': index
            },
            'inputs': {
                'Input': [rnn_input_name],
                'PreState': [last_hidden],
                'WeightList': [
                    weight_list_0,
                    weight_list_1,
                    weight_list_2,
                    weight_list_3
                ]
            },
            'outputs': {'Out': [output_name]},
            'type': 'rnn_origin'
        }

        origin_var = {
            'name': output_name,
            'persistable': False,
            'shape': [input_shape[0], input_shape[1], weight_0_shape[0]]
        }
        ops.append(origin_op)
        vars[output_name] = origin_var

        for bat in range(batch):
            matmul_output_name = 'lstm_' + str(index) + '_' + str(bat) + '.tmp_matmul'
            cell_output_name = 'lstm_' + str(index) + '_' + str(bat) + '.tmp_c'
            hidden_output_name = 'lstm_' + str(index) + '_' + str(bat) + '.tmp_h'

            matmul_op = {
                'attrs': {
                    'input_axis': bat,
                    'state_axis': index if bat == 0 else 0,
                    'batch': batch,
                    'reverse': False if index % 2 == 0 else True
                },
                'inputs': {
                    'Input': [output_name],
                    'PreState': [last_hidden],
                    'WeightList': [weight_list_1]
                },
                'outputs': {'Out': [matmul_output_name]},
                'type': 'rnn_matmul'
            }

            matmul_var = {
                'name': matmul_output_name,
                'persistable': False,
                'shape': [1, 1, weight_0_shape[0]]
            }

            ops.append(matmul_op)
            vars[matmul_output_name] = matmul_var

            cell_op = {
                'attrs': {
                    'state_axis': index if bat == 0 else 0,
                    'hidden_size': hidden_size
                },
                'inputs': {
                    'X': [matmul_output_name],
                    'Y': [last_cell]
                },
                'outputs': {'Out': [cell_output_name]},
                'type': 'rnn_cell'
            }

            cell_var = {
                'name': cell_output_name,
                'persistable': False,
                'shape': [1, 1, weight_1_shape[1]]
            }

            ops.append(cell_op)
            vars[cell_output_name] = cell_var

            hidden_op = {
                'attrs': {
                    'state_axis': index if bat == 0 else 0,
                    'hidden_size': hidden_size
                },
                'inputs': {
                    'X': [matmul_output_name],
                    'Y': [last_cell]
                },
                'outputs': {'Out': [hidden_output_name]},
                'type': 'rnn_hidden'
            }

            hidden_var = {
                'name': hidden_output_name,
                'persistable': False,
                'shape': [1, 1, weight_1_shape[1]]
            }

            ops.append(hidden_op)
            vars[hidden_output_name] = hidden_var

            last_hidden = hidden_output_name
            last_cell = cell_output_name

        # concat
        if index % 2 == 1:

            concat_list = []
            concat_num = 0
            # concat forword and backword
            for bat in range(batch):
                x_input_name_0 = 'lstm_' + str(index - 1) + '_' + str(bat) + '.tmp_h'
                x_input_name_1 = 'lstm_' + str(index) + '_' + str(batch - bat - 1) + '.tmp_h'
                concat_output_name = 'lstm_' + str(index - 1) + '_' + str(bat) + '.tmp_concat'
                concat_op = {
                    'attrs': {
                        'axis': 2
                    },
                    'inputs': {
                        'X': [x_input_name_0, x_input_name_1]
                    },
                    'outputs': {'Out': [concat_output_name]},
                    'type': 'concat'
                }

                concat_var = {
                    'name': concat_output_name,
                    'persistable': False,
                    'shape': [1, 1, weight_1_shape[1] * 2]
                }
                ops.append(concat_op)
                vars[concat_output_name] = concat_var
                concat_list.append(bat)
                concat_num += 1

            concat_mul(index, concat_list, concat_num)

    # 删除rnn op
    del ops[rnn_index]
