#come from: https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/utils/fake_ptq.py
import os
import paddle
from paddle.fluid.framework import IrGraph
from paddle.framework import core
from paddle.static.quantization import QuantizationTransformPass, QuantizationTransformPassV2, AddQuantDequantPass, AddQuantDequantPassV2, QuantizationFreezePass, QuantWeightPass
from paddle.static.quantization import utils

try:
    from paddle.static.quantization import quant_config
    TRANSFORM_PASS_OP_TYPES = list(
        quant_config.SUPPORT_WEIGHT_QUANTIZATION_OP_DICT.keys())
    QUANT_DEQUANT_PASS_OP_TYPES = list(
        quant_config.SUPPORT_ACT_QUANTIZATION_OP_DICT.keys())
except:
    TRANSFORM_PASS_OP_TYPES = utils._weight_supported_quantizable_op_type
    QUANT_DEQUANT_PASS_OP_TYPES = utils._act_supported_quantizable_op_type

from paddle.static import load_inference_model


def post_quant_fake(executor,
                    model_dir,
                    model_filename=None,
                    params_filename=None,
                    save_model_path=None,
                    quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
                    is_full_quantize=False,
                    activation_bits=8,
                    weight_bits=8,
                    onnx_format=False):
    """
    Utilizing post training quantization methon to quantize the FP32 model,
    and it not uses calibrate data and the fake model cannot be used in practice.
    Usage:
        paddle.enable_static()
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        post_quant_fake(executor=exe,
            model_dir='./inference_model/MobileNet/',
            model_filename='model',
            params_filename='params',
            save_model_path='fake_quant')
    """
    activation_quantize_type = 'range_abs_max'
    weight_quantize_type = 'channel_wise_abs_max'
    _dynamic_quantize_op_type = ['lstm']
    _weight_supported_quantizable_op_type = TRANSFORM_PASS_OP_TYPES
    _act_supported_quantizable_op_type = QUANT_DEQUANT_PASS_OP_TYPES
    _support_quantize_op_type = list(
        set(_weight_supported_quantizable_op_type +
            _act_supported_quantizable_op_type + _dynamic_quantize_op_type))
    _place = executor.place
    _scope = paddle.static.Scope()

    with paddle.static.scope_guard(_scope):
        if is_full_quantize:
            _quantizable_op_type = _support_quantize_op_type
        else:
            _quantizable_op_type = quantizable_op_type
            for op_type in _quantizable_op_type:
                assert op_type in _support_quantize_op_type, \
                    op_type + " is not supported for quantization."
        _program, _feed_list, _fetch_list = load_inference_model(
            model_dir,
            executor,
            model_filename=model_filename,
            params_filename=params_filename)

        graph = IrGraph(core.Graph(_program.desc), for_test=True)

        # use QuantizationTransformPass to insert fake_quant/fake_dequantize op
        major_quantizable_op_types = []
        for op_type in _weight_supported_quantizable_op_type:
            if op_type in _quantizable_op_type:
                major_quantizable_op_types.append(op_type)
        if onnx_format:
            transform_pass = QuantizationTransformPassV2(
                scope=_scope,
                place=_place,
                weight_bits=weight_bits,
                activation_bits=activation_bits,
                activation_quantize_type=activation_quantize_type,
                weight_quantize_type=weight_quantize_type,
                quantizable_op_type=major_quantizable_op_types)
        else:
            transform_pass = QuantizationTransformPass(
                scope=_scope,
                place=_place,
                weight_bits=weight_bits,
                activation_bits=activation_bits,
                activation_quantize_type=activation_quantize_type,
                weight_quantize_type=weight_quantize_type,
                quantizable_op_type=major_quantizable_op_types)

        for sub_graph in graph.all_sub_graphs():
            # Insert fake_quant/fake_dequantize op must in test graph, so
            # set per graph's _for_test is True.
            sub_graph._for_test = True
            transform_pass.apply(sub_graph)

        # use AddQuantDequantPass to insert fake_quant_dequant op
        minor_quantizable_op_types = []
        for op_type in _act_supported_quantizable_op_type:
            if op_type in _quantizable_op_type:
                minor_quantizable_op_types.append(op_type)
        if onnx_format:
            add_quant_dequant_pass = AddQuantDequantPassV2(
                scope=_scope,
                place=_place,
                quantizable_op_type=minor_quantizable_op_types)
        else:
            add_quant_dequant_pass = AddQuantDequantPass(
                scope=_scope,
                place=_place,
                quantizable_op_type=minor_quantizable_op_types)

        for sub_graph in graph.all_sub_graphs():
            sub_graph._for_test = True
            add_quant_dequant_pass.apply(sub_graph)

        # apply QuantizationFreezePass, and obtain the final quant model
        if onnx_format:
            quant_weight_pass = QuantWeightPass(_scope, _place)
            for sub_graph in graph.all_sub_graphs():
                sub_graph._for_test = True
            quant_weight_pass.apply(sub_graph)
        else:
            freeze_pass = QuantizationFreezePass(
                scope=_scope,
                place=_place,
                weight_bits=weight_bits,
                activation_bits=activation_bits,
                weight_quantize_type=weight_quantize_type,
                quantizable_op_type=major_quantizable_op_types)

            for sub_graph in graph.all_sub_graphs():
                sub_graph._for_test = True
                freeze_pass.apply(sub_graph)

        _program = graph.to_program()

        def save_info(op_node, out_var_name, out_info_name, quantized_type):
            op_node._set_attr(out_info_name, 0.001)
            op_node._set_attr("with_quant_attr", True)
            if op_node.type in _quantizable_op_type:
                op._set_attr("quantization_type", quantized_type)

        def analysis_and_save_info(op_node, out_var_name):
            argname_index = utils._get_output_name_index(op_node, out_var_name)
            assert argname_index is not None, \
                out_var_name + " is not the output of the op"

            save_info(op_node, out_var_name, "out_threshold", "post_avg")
            save_info(op_node, out_var_name,
                      argname_index[0] + str(argname_index[1]) + "_threshold",
                      "post_avg")

        for block_id in range(len(_program.blocks)):
            for op in _program.blocks[block_id].ops:
                if op.type in (
                        _quantizable_op_type +
                        list(quant_config.SUPPORT_QUANTIZATION_OP_DICT.keys())):
                    out_var_names = utils._get_op_output_var_names(op)
                    for var_name in out_var_names:
                        analysis_and_save_info(op, var_name)

        feed_vars = [_program.global_block().var(name) for name in _feed_list]
        model_name = model_filename.split('.')[
            0] if model_filename is not None else 'model'
        save_model_path = os.path.join(save_model_path, model_name)
        paddle.static.save_inference_model(
            path_prefix=save_model_path,
            feed_vars=feed_vars,
            fetch_vars=_fetch_list,
            executor=executor,
            program=_program,
            clip_extra=False)
        print("The quantized model is saved in: " + save_model_path)
