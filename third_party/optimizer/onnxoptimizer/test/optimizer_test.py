# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict
from typing import Sequence, Text, Any, Tuple, List, Callable, Optional, Dict, Union
import io
import unittest
import os

import numpy as np  # type: ignore

try:
    import torch
    import torchvision as tv
    has_tv = True
except:
    has_tv = False

import onnx
from onnx import checker, helper, ModelProto, TensorProto, GraphProto, NodeProto, shape_inference
from onnx import numpy_helper
from onnx.numpy_helper import to_array
try:
    import onnxruntime as rt
    has_ort = True
except:
    has_ort = False

import onnxoptimizer


TensorShape = List[int]
TensorShapes = Dict[Optional[str], TensorShape]

LATEST_STABLE_OPSET_VERSION = 13


class TestOptimizer(unittest.TestCase):
    def _compare(self, model_opt: onnx.ModelProto, model_ori: onnx.ModelProto, n_times: int = 5,
                 input_shapes: Optional[TensorShapes] = None, verbose=True) -> bool:
        """
        :param input_shapes: Shapes of generated random inputs
        :param model_opt: The simplified ONNX model
        :param model_ori: The original ONNX model
        :param n_times: Generate n random inputs
        """

        def get_shape_from_value_info_proto(v: onnx.ValueInfoProto) -> List[int]:
            return [dim.dim_value for dim in v.type.tensor_type.shape.dim]

        def get_value_info_all(m: onnx.ModelProto, name: str) -> Optional[onnx.ValueInfoProto]:
            for v in m.graph.value_info:
                if v.name == name:
                    return v

            for v in m.graph.input:
                if v.name == name:
                    return v

            for v in m.graph.output:
                if v.name == name:
                    return v

            return None

        def get_shape(m: onnx.ModelProto, name: str) -> TensorShape:
            """
            Note: This method relies on onnx shape inference, which is not reliable. So only use it on input or output tensors
            """
            v = get_value_info_all(m, name)
            if v is not None:
                return get_shape_from_value_info_proto(v)
            raise RuntimeError('Cannot get shape of "{}"'.format(name))

        def get_elem_type(m: onnx.ModelProto, name: str) -> Optional[int]:
            v = get_value_info_all(m, name)
            if v is not None:
                return v.type.tensor_type.elem_type
            return None

        def get_np_type_from_elem_type(elem_type: int) -> int:
            sizes = (None, np.float32, np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64, str, np.bool,
                     np.float16, np.double, np.uint32, np.uint64, np.complex64, np.complex128, np.float16)
            assert len(sizes) == 17
            size = sizes[elem_type]
            assert size is not None
            return size

        def get_input_names(model: onnx.ModelProto) -> List[str]:
            input_names = list(set([ipt.name for ipt in model.graph.input])
                               - set([x.name for x in model.graph.initializer]))
            return input_names

        def generate_rand_input(model, input_shapes: Optional[TensorShapes] = None):
            if input_shapes is None:
                input_shapes = {}
            input_names = get_input_names(model)
            full_input_shapes = {ipt: get_shape(
                model, ipt) for ipt in input_names}
            assert None not in input_shapes
            full_input_shapes.update(input_shapes)  # type: ignore
            for key in full_input_shapes:
                if np.prod(full_input_shapes[key]) <= 0:
                    raise RuntimeError(
                        'The shape of input "{}" has dynamic size, '
                        'please set an input shape manually'.format(key))

            inputs = {ipt: np.array(np.random.rand(*full_input_shapes[ipt]),
                                    dtype=get_np_type_from_elem_type(get_elem_type(model, ipt))) for ipt in
                      input_names}
            return inputs

        def forward(model, inputs=None, input_shapes: Optional[TensorShapes] = None) -> Dict[str, np.ndarray]:
            if input_shapes is None:
                input_shapes = {}
            sess_options = rt.SessionOptions()
            sess_options.graph_optimization_level = rt.GraphOptimizationLevel(0)
            sess_options.log_severity_level = 3
            sess = rt.InferenceSession(model.SerializeToString(
            ), sess_options=sess_options, providers=['CPUExecutionProvider'])
            if inputs is None:
                inputs = generate_rand_input(model, input_shapes=input_shapes)
            outputs = [x.name for x in sess.get_outputs()]
            run_options = rt.RunOptions()
            run_options.log_severity_level = 3
            res = OrderedDict(zip(outputs, sess.run(
                outputs, inputs, run_options=run_options)))
            return res

        if input_shapes is None:
            input_shapes = {}
        onnx.checker.check_model(model_opt)
        for i in range(n_times):
            rand_input = generate_rand_input(
                model_opt, input_shapes=input_shapes)
            res_ori = forward(model_ori, inputs=rand_input)
            res_opt = forward(model_opt, inputs=rand_input)

            for name in res_opt.keys():
                if not np.allclose(res_opt[name], res_ori[name], rtol=1e-4, atol=1e-5):
                    if verbose:
                        print("Tensor {} changes after optimization. The max diff is {}.".format(
                            name, np.max(np.abs(res_opt[name] - res_ori[name]))))
                        print("After optimization:")
                        print(res_opt[name])
                        print("Before optimization:")
                        print(res_ori[name])
                        print("----------------")
                    return False
        return True

    # type: (Union[GraphProto, ModelProto], Sequence[Text], bool, **Any) -> ModelProto
    def _optimized(self, graph_or_model, opts, fixed_point=False, compare_result=True, check=True, **kwargs):
        if compare_result and not check:
            self.fail("compare_result cannot be True if check is False")

        if isinstance(graph_or_model, ModelProto):
            orig_model = graph_or_model
        else:
            opset_imports = kwargs.pop('opset_imports', None)
            if opset_imports is None:
                opset_imports = [helper.make_opsetid(
                    "", LATEST_STABLE_OPSET_VERSION)]

            orig_model = helper.make_model(
                graph_or_model, producer_name='onnx-test', opset_imports=opset_imports, **kwargs)
        checker.check_model(orig_model)
        optimized_model = onnxoptimizer.optimize(orig_model, opts, fixed_point)
        # NOTE(daquexian): Some passes (like lift_lexical_references) generate illegal model intentionally
        if check:
            checker.check_model(optimized_model)
        if compare_result and len(optimized_model.graph.node) > 0:
            if has_ort:
                assert self._compare(optimized_model, orig_model)
            else:
                print("Skip onnxruntime test because it is not installed.")
        return optimized_model

    # input_types and output_types are lists of triples of (name, type, shape)
    # NOTE(daquexian): only values that change across loop iterations should be in `input_types` and `output_types`. The pseudocode showing how loop op works is:
    # loop_value_inputs = graph_value_inputs
    # while cond:
    #   loop_value_outputs = body(loop_value_inputs)
    #   loop_value_inputs = loop_value_outputs
    # graph_value_outputs = loop_value_outputs
    def _make_fake_loop_op(self,
                           body_nodes,   # type: Sequence[NodeProto]
                           # type: Sequence[Tuple[TensorProto.DataType, Sequence[int], Text]]
                           input_types,
                           # type: Sequence[Tuple[TensorProto.DataType, Sequence[int], Text]]
                           output_types,
                           check_legality=True,
                           ):  # type: (...) -> List[NodeProto]
        if check_legality:
            assert len(input_types) == len(output_types)
        zero = helper.make_tensor(
            "trip_count_value", TensorProto.INT64, (), [1])
        true = helper.make_tensor("condition", TensorProto.BOOL, (), [True])
        # lcd is a dummy loop-carried dependency that only exists because
        # right now the schema checker is broken and assumes a variadic
        # input needs at least one value.
        graph_inputs = [helper.make_tensor_value_info("i", TensorProto.INT64, ()),
                        helper.make_tensor_value_info("cond", TensorProto.BOOL, ())]
        for type, shape, name in input_types:
            graph_inputs.append(
                helper.make_tensor_value_info("_" + name, type, shape))
        graph_outputs = [helper.make_tensor_value_info(
            "cond", TensorProto.BOOL, ())]
        for type, shape, name in output_types:
            graph_outputs.append(
                helper.make_tensor_value_info("_" + name, type, shape))
        body_graph = helper.make_graph(body_nodes, "body_graph", graph_inputs,
                                       graph_outputs)
        loop_inputs = ["trip_count", "condition"]
        loop_inputs.extend([name for _, _, name in input_types])
        # TODO: fix checker to accept 0-input variadic inputs
        if len(loop_inputs) == 2:
            loop_inputs.append("")
        loop_outputs = [name for _, _, name in output_types]
        retval_nodes = [
            helper.make_node("Constant", [], ["trip_count"], value=zero),
            helper.make_node("Constant", [], ["condition"], value=true),
            helper.make_node("Loop", loop_inputs, loop_outputs, body=body_graph)
        ]
        return retval_nodes

    def _make_fake_if_op(self,
                         true_nodes,   # type: Sequence[NodeProto]
                         false_nodes,  # type: Sequence[NodeProto]
                         # type: Sequence[Tuple[TensorProto.DataType, Sequence[int], Text]]
                         output_types
                         ):  # type: (...) -> List[NodeProto]
        true = helper.make_tensor("condition", TensorProto.BOOL, (), [True])
        true_graph = helper.make_graph(true_nodes, "true_graph", [], [])
        false_graph = helper.make_graph(false_nodes, "false_graph", [], [])
        if_inputs = ["condition"]
        if_outputs = [name for _, _, name in output_types]
        retval_nodes = [
            helper.make_node("Constant", [], ["condition"], value=true),
            helper.make_node("If", if_inputs, if_outputs, then_branch=true_graph,
                             else_branch=false_graph)
        ]
        return retval_nodes

    # fn is a function that takes a single node as argument
    # type: (GraphProto, Callable[[NodeProto], None]) -> None
    def _visit_all_nodes_recursive(self, graph, fn):
        for node in graph.node:
            fn(node)
            for attr in node.attribute:
                if attr.g is not None:
                    self._visit_all_nodes_recursive(attr.g, fn)
                if len(attr.graphs):
                    for gr in attr.graphs:
                        self._visit_all_nodes_recursive(gr, fn)

    def test_get_available_passes(self):  # type: () -> None
        # FIXME does not guarantees to be listing all
        graph = helper.make_graph([], "dummy_graph", [], [])
        list_of_passes = onnxoptimizer.get_available_passes()
        assert isinstance(list_of_passes, (list)) and len(list_of_passes) > 0
        for pass_name in list_of_passes:
            # If pass_name is invalid it throws a RuntimeError
            self._optimized(graph, [pass_name])

    def test_eliminate_identity_single_use(self):  # type: () -> None
        nodes = [helper.make_node("Add", ["X", "Y"], ["A"]),
                 helper.make_node("Identity", ["A"], ["B"])]
        nodes.extend(self._make_fake_loop_op(
            [helper.make_node("Relu", ["_B"], ["_B1"]),
             helper.make_node("Identity", ["_B1"], ["_B2"])],
            [(TensorProto.FLOAT, (5,), "B")],
            [(TensorProto.FLOAT, (5,), "B2")]))
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (5,)),
             helper.make_tensor_value_info("B2", TensorProto.FLOAT, (5,))])
        optimized_model = self._optimized(graph, ["eliminate_identity"])

        # All identity nodes should have been eliminated
        def check_identity(node):  # type: (NodeProto) -> None
            assert node.op_type != "Identity"
        self._visit_all_nodes_recursive(optimized_model.graph, check_identity)
        # Use of the output from the Identity node in the main graph should
        # have been replaced with the input to the identity node
        assert len(optimized_model.graph.output) == 2
        assert optimized_model.graph.output[0].name == "B"
        # Use of the output from the Identity node in the loop graph should
        # have been replaced with the input to that identity node
        assert len(optimized_model.graph.node[3].attribute[0].g.output) == 2
        assert optimized_model.graph.node[3].attribute[0].g.output[1].name == "_B2"

    # type: () -> None
    def test_eliminate_identity_both_graph_input_and_output(self):
        # We should not eliminate an op when its input is also graph input,
        # and its output is also graph output, because we want to always keep
        # the name of graph input and output unchanged.
        identity = helper.make_node("Identity", ["A"], ["B"])
        graph = helper.make_graph(
            [identity],
            "test",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (5,))],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (5,))])
        optimized_model = self._optimized(graph, ["eliminate_identity"])

        assert optimized_model.graph == graph

    def test_eliminate_if_with_const_cond(self):  # type: () -> None
        true = helper.make_tensor("condition", TensorProto.BOOL, (), [True])

        subgraph_output_info = helper.make_tensor_value_info(
            "C", TensorProto.FLOAT, (5,))

        sin = helper.make_node("Sin", ["A"], ["B"])
        hard_sigmoid = helper.make_node(
            "HardSigmoid", ["B"], ["C"], alpha=0.4, beta=0.6)
        true_graph = helper.make_graph(
            [sin, hard_sigmoid], "true_graph", [], [subgraph_output_info])

        identity = helper.make_node("Identity", ["A"], ["C"])
        false_graph = helper.make_graph(
            [identity], "false_graph", [], [subgraph_output_info])

        graph = helper.make_graph([
            helper.make_node("Constant", [], ["condition"], value=true),
            helper.make_node("If", ["condition"], ["result"], then_branch=true_graph,
                             else_branch=false_graph)],
            "test",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (5,))],
            [helper.make_tensor_value_info("result", TensorProto.FLOAT, (5,))])
        optimized_model = self._optimized(
            graph, ["eliminate_if_with_const_cond"])
        assert len(optimized_model.graph.node) == 3
        assert optimized_model.graph.node[0].op_type == "Constant"
        assert optimized_model.graph.node[1].op_type == "Sin"
        assert optimized_model.graph.node[2].op_type == "HardSigmoid"

    def test_eliminate_identity_graph_output(self):  # type: () -> None
        add = helper.make_node("Add", ["X", "Y"], ["A"])
        identity = helper.make_node("Identity", ["A"], ["B"])
        graph = helper.make_graph(
            [add, identity],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (5,))])
        optimized_model = self._optimized(graph, ["eliminate_identity"])

        for node in optimized_model.graph.node:
            assert node.op_type != "Identity"
        assert len(
            optimized_model.graph.output) == 1 and optimized_model.graph.output[0].name == 'B'
        assert len(optimized_model.graph.node) == 1

    def test_eliminate_identity_multiple_uses(self):  # type: () -> None
        identity = helper.make_node("Identity", ["X"], ["Y"])
        add = helper.make_node("Add", ["Z", "Y"], ["A"])
        mul = helper.make_node("Mul", ["A", "Y"], ["B"])
        graph = helper.make_graph(
            [identity, add, mul],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,)),
             helper.make_tensor_value_info("Z", TensorProto.FLOAT, (5,))],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (5,))])
        optimized_model = self._optimized(graph, ["eliminate_identity"])

        for node in optimized_model.graph.node:
            assert node.op_type != "Identity"
        assert len(optimized_model.graph.node) == 2

    def test_not_fuse_non_nop_flatten(self):
        identity = helper.make_node("Identity", ["A"], ["X"])
        flatten = helper.make_node("Flatten", ["X"], ["B"], axis=2)
        graph = helper.make_graph(
            [identity, flatten],
            "test",
            [helper.make_tensor_value_info(
                "A", TensorProto.FLOAT, (1, 10, 3, 1, 1))],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (10, 3))])

        optimized_model = self._optimized(graph, ["eliminate_nop_flatten"])

        assert len(optimized_model.graph.node) == 2
        assert optimized_model.graph.node[0].op_type == 'Identity'
        assert optimized_model.graph.node[1].op_type == 'Flatten'

    def test_nop_flatten_axis0_graph_output(self):
        add = helper.make_node("Add", ["X", "Y"], ["A"])
        flatten = helper.make_node("Flatten", ["A"], ["B"], axis=0)
        graph = helper.make_graph(
            [add, flatten],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 10)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (1, 10)),
             ],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (1, 10))],
            # the tensor_value_info of "A" is necessary to this optimizer
            value_info=[helper.make_tensor_value_info(
                "A", TensorProto.FLOAT, (1, 10))]
        )
        # The existence of shape infos of graoh outputs is checked in _optimized
        optimized_model = self._optimized(graph, ["eliminate_nop_flatten"])

        assert len(optimized_model.graph.node) == 1
        assert optimized_model.graph.node[0].op_type == 'Add'

    def test_nop_flatten_axis0(self):
        identity = helper.make_node("Identity", ["A"], ["X"])
        flatten = helper.make_node("Flatten", ["X"], ["B"], axis=0)
        graph = helper.make_graph(
            [identity, flatten],
            "test",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (1, 10))],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (1, 10))],
            value_info=[helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 10))])

        optimized_model = self._optimized(graph, ["eliminate_nop_flatten"])

        assert len(optimized_model.graph.node) == 1
        assert optimized_model.graph.node[0].op_type == "Identity"

    def test_nop_flatten_axis1(self):
        identity = helper.make_node("Identity", ["A"], ["X"])
        flatten = helper.make_node("Flatten", ["X"], ["B"], axis=1)
        graph = helper.make_graph(
            [identity, flatten],
            "test",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (2, 3))],
            value_info=[helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3))])

        optimized_model = self._optimized(graph, ["eliminate_nop_flatten"])

        assert len(optimized_model.graph.node) == 1
        assert optimized_model.graph.node[0].op_type == "Identity"

    def test_eliminate_duplicate_initializer(self):  # type: () -> None
        add_1 = helper.make_node("Add", ["A", "I_0"], ["B"])
        add_2 = helper.make_node("Add", ["B", "I_1"], ["C"])
        i = np.random.rand(5).astype(np.float32)
        graph = helper.make_graph(
            [add_1, add_2],
            "test",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (5,))],
            [helper.make_tensor_value_info("C", TensorProto.FLOAT, (5,))],
            [helper.make_tensor("I_0", TensorProto.FLOAT,
                                dims=(5,),
                                vals=i.tobytes(),
                                raw=True),
             helper.make_tensor("I_1", TensorProto.FLOAT,
                                dims=(5,),
                                vals=i.tobytes(),
                                raw=True)])
        optimized_model = self._optimized(
            graph, ["eliminate_duplicate_initializer"])
        assert len(optimized_model.graph.node) == 2
        assert len(optimized_model.graph.initializer) == 1
        assert len(optimized_model.graph.input) == 1
        assert optimized_model.graph.node[0].input[1] == "I_0"

    def test_nop_cast(self):  # type: () -> None
        identity = helper.make_node("Identity", ["X"], ["A"])
        cast = helper.make_node("Cast", ["A"], ["B"], to=TensorProto.FLOAT)
        graph = helper.make_graph(
            [identity, cast],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (2, 3))],
            value_info=[helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3))])

        optimized_model = self._optimized(graph, ["eliminate_nop_cast"])

        assert len(optimized_model.graph.node) == 1
        assert optimized_model.graph.node[0].op_type == "Identity"

    def test_nop_transpose_graph_output(self):  # type: () -> None
        add = helper.make_node("Add", ["X", "Y"], ["A"])
        trans = helper.make_node("Transpose", ["A"], ["B"], perm=[0, 1])
        graph = helper.make_graph(
            [add, trans],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (2, 3))])
        # The existence of shape infos of graoh outputs is checked in _optimized
        optimized_model = self._optimized(graph, ["eliminate_nop_transpose"])

        def check_transpose(node):  # type: (NodeProto) -> None
            assert node.op_type != "Transpose"
        self._visit_all_nodes_recursive(optimized_model.graph, check_transpose)
        assert len(optimized_model.graph.node) == 1

    def test_nop_transpose(self):  # type: () -> None
        nodes = [helper.make_node("Identity", ["A"], ["X"]),
                 helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 1])]
        nodes.extend(self._make_fake_loop_op(
            [helper.make_node("Identity", ["_Y"], ["Y1"]),
             helper.make_node("Transpose", ["Y1"], ["_Y2"], perm=[0, 1])],
            [(TensorProto.FLOAT, (2, 3), "Y")],
            [(TensorProto.FLOAT, (2, 3), "Y2")]))
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 3)),
             helper.make_tensor_value_info("Y2", TensorProto.FLOAT, (2, 3))],
            value_info=[helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3))])
        optimized_model = self._optimized(graph, ["eliminate_nop_transpose"])

        def check_transpose(node):  # type: (NodeProto) -> None
            assert node.op_type != "Transpose"
        self._visit_all_nodes_recursive(optimized_model.graph, check_transpose)
        # Use of the output from the Transpose node in the main graph should
        # have been replaced with the input to the identity node
        assert len(optimized_model.graph.output) == 2
        assert optimized_model.graph.output[0].name == "Y"
        # Use of the output from the Transpose node in the loop graph should
        # have been replaced with the input to that identity node
        assert len(optimized_model.graph.node[3].attribute[0].g.output) == 2
        assert optimized_model.graph.node[3].attribute[0].g.output[1].name == "_Y2"

    def test_nop_transpose_default(self):  # type: () -> None
        identity = helper.make_node("Identity", ["A"], ["X"])
        trans = helper.make_node("Transpose", ["X"], ["Y"])
        graph = helper.make_graph(
            [identity, trans],
            "test",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (3, 2))])
        optimized_model = self._optimized(graph, ["eliminate_nop_transpose"])

        assert optimized_model.graph == graph

    def test_nop_pad_opset10(self):  # type: () -> None
        identity = helper.make_node("Identity", ["A"], ["X"])
        pad = helper.make_node("Pad", ["X"], ["Y"], pads=[0, 0, 0, 0])
        graph = helper.make_graph(
            [identity, pad],
            "test",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 3))])
        optimized_model = self._optimized(
            graph, ["eliminate_nop_pad"], False, opset_imports=[helper.make_opsetid("", 10)])

        def check_pad(node):  # type: (NodeProto) -> None
            assert node.op_type != "Pad"
        self._visit_all_nodes_recursive(optimized_model.graph, check_pad)
        assert len(optimized_model.graph.output) == 1
        assert optimized_model.graph.output[0].name == "Y"
        assert len(optimized_model.graph.node) == 1

    def test_nop_pad_graph_output(self):  # type: () -> None
        add = helper.make_node("Add", ["X", "Y"], ["A"])
        pad = helper.make_node("Pad", ["A", "Pads"], ["B"])
        graph = helper.make_graph(
            [add, pad],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,)),
             helper.make_tensor_value_info("Pads", TensorProto.INT64, (2,))],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (5,))],
            [helper.make_tensor("Pads", TensorProto.INT64,
                                dims=(2,),
                                vals=np.array([0, 0]).astype(
                                    np.int64).tobytes(),
                                raw=True)])
        # The existence of shape infos of graoh outputs is checked in _optimized
        optimized_model = self._optimized(graph, ["eliminate_nop_pad"])

        def check_pad(node):  # type: (NodeProto) -> None
            assert node.op_type != "Pad"
        self._visit_all_nodes_recursive(optimized_model.graph, check_pad)
        assert len(optimized_model.graph.node) == 1

    def test_nop_pad(self):  # type: () -> None
        identity = helper.make_node("Identity", ["A"], ["X"])
        pad = helper.make_node("Pad", ["X", "Pads"], ["Y"])
        graph = helper.make_graph(
            [identity, pad],
            "test",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3)),
             helper.make_tensor_value_info("Pads", TensorProto.INT64, (4,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor("Pads", TensorProto.INT64,
                                dims=(4,),
                                vals=np.array([0, 0, 0, 0]).astype(
                                    np.int64).tobytes(),
                                raw=True)])
        optimized_model = self._optimized(graph, ["eliminate_nop_pad"])

        def check_pad(node):  # type: (NodeProto) -> None
            assert node.op_type != "Pad"
        self._visit_all_nodes_recursive(optimized_model.graph, check_pad)
        assert len(optimized_model.graph.output) == 1
        assert optimized_model.graph.output[0].name == "Y"
        assert len(optimized_model.graph.node) == 1

    def test_nop_pad_default_opset10(self):  # type: () -> None
        identity = helper.make_node("Identity", ["A"], ["X"])
        pad = helper.make_node("Pad", ["X"], ["Y"], pads=[0, 0, 1, 1])
        graph = helper.make_graph(
            [identity, pad],
            "test",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 4))])
        optimized_model = self._optimized(
            graph, ["eliminate_nop_pad"], False, opset_imports=[helper.make_opsetid("", 10)])

        assert len(list(optimized_model.graph.node)) == 2
        assert optimized_model.graph == graph

    def test_nop_pad_default(self):  # type: () -> None
        identity = helper.make_node("Identity", ["A"], ["X"])
        pad = helper.make_node("Pad", ["X", "Pads"], ["Y"])
        graph = helper.make_graph(
            [identity, pad],
            "test",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3)),
             helper.make_tensor_value_info("Pads", TensorProto.INT64, (4,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 4))],
            [helper.make_tensor("Pads", TensorProto.INT64,
                                dims=(4,),
                                vals=np.array([0, 1, 0, 0]).astype(
                                    np.int64).tobytes(),
                                raw=True)])
        optimized_model = self._optimized(graph, ["eliminate_nop_pad"])

        assert len(list(optimized_model.graph.node)) == 2
        assert optimized_model.graph == graph

    def test_eliminate_unused_initializer(self):  # type: () -> None
        add = helper.make_node("Add", ["X", "Y"], ["Z"])
        graph = helper.make_graph(
            [add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 2)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (1, 2))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 2))],
            [helper.make_tensor("A", TensorProto.FLOAT,
                                dims=(2, 3),
                                vals=np.random.randn(2, 3).astype(
                                    np.float32).tobytes(),
                                raw=True)])
        optimized_model = self._optimized(
            graph, ["eliminate_unused_initializer"])

        assert len(list(optimized_model.graph.initializer)) == 0

    def test_eliminate_unused_initializer_input(self):  # type: () -> None
        add = helper.make_node("Add", ["X", "Y"], ["Z"])
        graph = helper.make_graph(
            [add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 2)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (1, 2)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 2))],
            [helper.make_tensor("A", TensorProto.FLOAT,
                                dims=(2, 3),
                                vals=np.random.randn(2, 3).astype(
                                    np.float32).tobytes(),
                                raw=True)])
        optimized_model = self._optimized(
            graph, ["eliminate_unused_initializer"])

        assert len(list(optimized_model.graph.initializer)) == 0
        assert len(optimized_model.graph.input) == 2

    # type: () -> None
    def test_eliminate_unused_initializer_no_eliminate_used_default(self):
        add = helper.make_node("Add", ["X", "A"], ["Z"])
        graph = helper.make_graph(
            [add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 2)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (1, 2))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 2))],
            [helper.make_tensor("A", TensorProto.FLOAT,
                                dims=(1, 2),
                                vals=np.random.randn(1, 2).astype(
                                    np.float32).tobytes(),
                                raw=True)])
        optimized_model = self._optimized(
            graph, ["eliminate_unused_initializer"])

        assert len(list(optimized_model.graph.initializer)) == 1

    # type: () -> None
    def test_eliminate_unused_initializer_no_eliminate_used(self):
        nodes = [helper.make_node("Add", ["X", "A"], ["Z"])]
        nodes.extend(self._make_fake_loop_op(
            [helper.make_node("Add", ["_X", "A"], ["_Z2"])],
            [(TensorProto.FLOAT, (1, 2), "X")],
            [(TensorProto.FLOAT, (1, 2), "Z2")]))
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 2)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (1, 2))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 2))],
            [helper.make_tensor("A", TensorProto.FLOAT,
                                dims=(1, 2),
                                vals=np.random.randn(1, 2).astype(
                                    np.float32).tobytes(),
                                raw=True)])
        optimized_model = self._optimized(
            graph, ["eliminate_unused_initializer"])

        # Add, Constant (trip count), Constant (cond), Loop
        assert len(list(optimized_model.graph.node)) == 4
        assert optimized_model.graph.node[0].op_type == "Add"
        assert optimized_model.graph.output[0].name == "Z"
        # Add
        assert len(optimized_model.graph.node[3].attribute[0].g.node) == 1
        assert optimized_model.graph.node[3].attribute[0].g.node[0].op_type == 'Add'
        assert optimized_model.graph.node[3].attribute[0].g.output[1].name == '_Z2'

        assert len(list(optimized_model.graph.initializer)) == 1

    # type: () -> None
    def test_eliminate_unused_initializer_no_eliminate_output(self):
        add = helper.make_node("Add", ["X", "Y"], ["Z"])
        graph = helper.make_graph(
            [add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 2)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (1, 2)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 2)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor("A", TensorProto.FLOAT,
                                dims=(2, 3),
                                vals=np.random.randn(2, 3).astype(
                                    np.float32).tobytes(),
                                raw=True)])
        optimized_model = self._optimized(
            graph, ["eliminate_unused_initializer"])

        assert len(list(optimized_model.graph.initializer)) == 1
        assert "Z" in [o.name for o in optimized_model.graph.output]

    def test_extract_constant_to_initializer(self):  # type: () -> None
        conv = helper.make_node("Conv", ["X", "Y"], ["Z"])
        constant = helper.make_node("Constant", [], ["A"],
                                    value=helper.make_tensor(
                                        name="bias",
                                        data_type=TensorProto.FLOAT,
                                        dims=(16, 1, 1),
                                        vals=np.random.randn(16).astype(np.float32).tolist()))
        add = helper.make_node("Add", ["Z", "A"], ["B"])
        graph = helper.make_graph(
            [conv, constant, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3))],
            [helper.make_tensor_value_info(
                "B", TensorProto.FLOAT, (1, 16, 1, 1))],
        )
        optimized_model = self._optimized(
            graph, ["extract_constant_to_initializer"])

        self.assertEqual(len(optimized_model.graph.initializer), 1)
        init = optimized_model.graph.initializer[0]
        self.assertEqual(init.name, 'A')
        self.assertEqual(init.dims, [16, 1, 1])
        self.assertEqual(init.data_type, TensorProto.FLOAT)

        self.assertEqual(
            [n.op_type for n in optimized_model.graph.node], ['Conv', 'Add'])

    def test_fuse_concats(self):  # type: () -> None
        nodes = [helper.make_node("Concat", ["A", "B", "C"], ["X"], axis=0),
                 helper.make_node("Concat", ["D", "E", "F"], ["Y"], axis=0),
                 helper.make_node("Concat", ["X", "G", "Y"], ["Z"], axis=0)]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3, 4)),
             helper.make_tensor_value_info("B", TensorProto.FLOAT, (4, 3, 4)),
             helper.make_tensor_value_info("C", TensorProto.FLOAT, (2, 3, 4)),
             helper.make_tensor_value_info("D", TensorProto.FLOAT, (4, 3, 4)),
             helper.make_tensor_value_info("E", TensorProto.FLOAT, (2, 3, 4)),
             helper.make_tensor_value_info("F", TensorProto.FLOAT, (4, 3, 4)),
             helper.make_tensor_value_info("G", TensorProto.FLOAT, (4, 3, 4))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (22, 3, 4))])
        optimized_model = self._optimized(
            graph, ["fuse_consecutive_concats"], True)  # two passes are needed to simplify the graph to its simplest state.

        assert len(optimized_model.graph.node) == 1
        assert len(optimized_model.graph.node[0].input) == 7
        assert optimized_model.graph.node[0].input == [
            "A", "B", "C", "G", "D", "E", "F"]
        assert optimized_model.graph.node[0].op_type == "Concat"

    def test_fuse_concats_different_axis(self):  # type: () -> None
        nodes = [helper.make_node("Concat", ["A", "B", "C"], ["X"], axis=0),
                 helper.make_node("Concat", ["D", "E", "F"], ["Y"], axis=1),
                 helper.make_node("Concat", ["X", "Y"], ["Z"], axis=2)]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 9, 4)),
             helper.make_tensor_value_info("B", TensorProto.FLOAT, (4, 9, 4)),
             helper.make_tensor_value_info("C", TensorProto.FLOAT, (2, 9, 4)),
             helper.make_tensor_value_info("D", TensorProto.FLOAT, (8, 3, 4)),
             helper.make_tensor_value_info("E", TensorProto.FLOAT, (8, 3, 4)),
             helper.make_tensor_value_info("F", TensorProto.FLOAT, (8, 3, 4))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (8, 9, 8))])
        optimized_model = self._optimized(
            graph, ["fuse_consecutive_concats"])

        assert optimized_model.graph == graph

    def test_fuse_transpose(self):  # type: () -> None
        nodes = [helper.make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2]),
                 helper.make_node("Transpose", ["Y"], ["Z"], perm=[2, 0, 1]),
                 helper.make_node("Transpose", ["Z"], ["A"], perm=[2, 0, 1])]
        nodes.extend(self._make_fake_loop_op(
            [helper.make_node("Transpose", ["_X"], ["_Y2"], perm=[1, 0, 2]),
             helper.make_node("Transpose", ["_Y2"], ["_Y3"], perm=[2, 0, 1]),
             helper.make_node("Transpose", ["_Y3"], ["_Y4"], perm=[2, 0, 1])],
            [(TensorProto.FLOAT, (2, 3, 4), "X")],
            [(TensorProto.FLOAT, (2, 4, 3), "Y4")]))
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4))],
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 4, 3)),
             helper.make_tensor_value_info("Y4", TensorProto.FLOAT, (4, 3, 2))])
        original_model = helper.make_model(graph)
        shape_inference.infer_shapes(original_model)
        optimized_model = self._optimized(
            graph, ["fuse_consecutive_transposes"])
        shape_inference.infer_shapes(optimized_model)

        # Transpose, Constant (trip count), Constant (cond), Loop
        assert len(list(optimized_model.graph.node)) == 4
        # Transpose
        assert len(optimized_model.graph.node[3].attribute[0].g.node) == 1

    def test_fuse_transpose_default_graph_output(self):  # type: () -> None
        add = helper.make_node("Add", ["X", "Y"], ["A"])
        trans1 = helper.make_node("Transpose", ["A"], ["B"])
        trans2 = helper.make_node("Transpose", ["B"], ["C"])
        graph = helper.make_graph(
            [add, trans1, trans2],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("C", TensorProto.FLOAT, (2, 3))])
        # The existence of shape infos of graoh outputs is checked in _optimized
        optimized_model = self._optimized(
            graph, ["fuse_consecutive_transposes"])

        def check_transpose(node):  # type: (NodeProto) -> None
            assert node.op_type != "Transpose"
        self._visit_all_nodes_recursive(optimized_model.graph, check_transpose)
        assert len(optimized_model.graph.node) == 1

    def test_fuse_transpose_default(self):  # type: () -> None
        identity1 = helper.make_node("Identity", ["A"], ["X"])
        trans1 = helper.make_node("Transpose", ["X"], ["Y"])
        trans2 = helper.make_node("Transpose", ["Y"], ["Z"])
        identity2 = helper.make_node("Identity", ["Z"], ["B"])
        graph = helper.make_graph(
            [identity1, trans1, trans2, identity2],
            "test",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3, 4))],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (2, 3, 4))])
        optimized_model = self._optimized(
            graph, ["fuse_consecutive_transposes"])

        assert len(list(optimized_model.graph.node)) == 2
        assert optimized_model.graph.node[0].op_type == "Identity"
        assert optimized_model.graph.node[1].op_type == "Identity"

    def test_fuse_transpose_default_no_fuse(self):  # type: () -> None
        identity1 = helper.make_node("Identity", ["A"], ["X"])
        trans1 = helper.make_node("Transpose", ["X"], ["Y"])
        trans2 = helper.make_node("Transpose", ["Y"], ["Z"], perm=[0, 1, 2])
        identity2 = helper.make_node("Identity", ["Z"], ["B"])
        graph = helper.make_graph(
            [identity1, trans1, trans2, identity2],
            "test",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3, 4))],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (4, 3, 2))])
        optimized_model = self._optimized(
            graph, ["fuse_consecutive_transposes"])

        assert len(list(optimized_model.graph.node)) == 4
        assert optimized_model.graph == graph

    def test_fuse_transpose_into_gemm(self):  # type: () -> None
        nodes = [helper.make_node("Transpose", ["X"], ["A"], perm=[1, 0]),
                 helper.make_node("Transpose", ["Y"], ["B"], perm=[1, 0]),
                 helper.make_node("Gemm", ["A", "B", "C"], ["Z"])]
        nodes.extend(self._make_fake_loop_op(
            [helper.make_node("Transpose", ["_X"], ["_A"], perm=[1, 0]),
             helper.make_node("Transpose", ["Y"], ["_B"], perm=[1, 0]),
             helper.make_node("Gemm", ["_A", "_B", "C"], ["_Z2"])],
            [(TensorProto.FLOAT, (2, 3), "X")],
            [(TensorProto.FLOAT, (3, 5), "Z2")]))
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5, 2)),
             helper.make_tensor_value_info("C", TensorProto.FLOAT, (3, 5))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (3, 5))])
        optimized_model = self._optimized(graph, ["fuse_transpose_into_gemm"])

        # Gemm, Constant (trip count), Constant (cond), Loop
        assert len(list(optimized_model.graph.node)) == 4
        assert optimized_model.graph.node[0].op_type == "Gemm"
        # Gemm
        assert len(optimized_model.graph.node[3].attribute[0].g.node) == 1
        assert optimized_model.graph.node[3].attribute[0].g.node[0].op_type == "Gemm"

    def test_fuse_add_bias_into_conv_with_scalar_bias(self):  # type: () -> None
        nodes = [helper.make_node("Conv", ["X", "Y"], ["Z"]),
                 helper.make_node("Add", ["Z", "A"], ["B"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3)),
             helper.make_tensor_value_info(
                 "Y", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, ())],
            [helper.make_tensor_value_info(
                "B", TensorProto.FLOAT, (1, 16, 1, 1))],
        )
        optimized_model = self._optimized(graph, ["fuse_add_bias_into_conv"])

        # Unsqueeze, Conv
        assert len(optimized_model.graph.node) == 4
        assert optimized_model.graph.node[0].op_type == 'Unsqueeze'
        assert optimized_model.graph.node[1].op_type == 'Constant'
        assert optimized_model.graph.node[2].op_type == 'Tile'
        assert optimized_model.graph.node[3].op_type == 'Conv'

    def test_fuse_add_bias_into_conv_use_weight_shape(self):  # type: () -> None
        nodes = [helper.make_node("Conv", ["X", "Y"], ["Z"]),
                 helper.make_node("Add", ["Z", "A"], ["B"])]
        # FIXME(daquexian): It looks like subgraph cannot get value info from parent subgraph
        # nodes.extend(self._make_fake_loop_op(
        #     [helper.make_node("Conv", ["_X", "Y"], ["_Z"]),
        #      helper.make_node("Add", ["_Z", "A"], ["_B2"])],
        #     [(TensorProto.FLOAT, (1, 5, 3, 3), "X")],
        #     [(TensorProto.FLOAT, (1, 16, 1, 1), "B2")]))
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3)),
             helper.make_tensor_value_info(
                 "Y", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (16, 1, 1))],
            [helper.make_tensor_value_info(
                "B", TensorProto.FLOAT, (1, 16, 1, 1))],
        )
        optimized_model = self._optimized(graph, ["fuse_add_bias_into_conv"])

        # # Squeeze, Conv, Constant (trip count), Constant (condition), Loop
        # assert len(list(optimized_model.graph.node)) == 5
        assert len(list(optimized_model.graph.node)) == 2
        assert optimized_model.graph.node[0].op_type == 'Squeeze'
        assert optimized_model.graph.node[1].op_type == 'Conv'
        assert optimized_model.graph.output[0].name == 'B'
        # # Squeeze, Conv
        # assert len(optimized_model.graph.node[4].attribute[0].g.node) == 2
        # assert optimized_model.graph.node[4].attribute[0].g.node[0].op_type == 'Squeeze'
        # assert optimized_model.graph.node[4].attribute[0].g.node[1].op_type == 'Conv'
        # # Output 1 since 0 is 'cond'
        # assert optimized_model.graph.node[4].attribute[0].g.output[1].name == 'B2'

    # type: () -> None
    def test_fuse_add_bias_into_conv_use_weight_shape_with_tile(self):
        conv = helper.make_node("Conv", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "A"], ["B"])
        graph = helper.make_graph(
            [conv, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3)),
             helper.make_tensor_value_info(
                 "Y", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info(
                "B", TensorProto.FLOAT, (1, 16, 1, 1))],
        )
        optimized_model = self._optimized(graph, ["fuse_add_bias_into_conv"])

        assert len(list(optimized_model.graph.node)) == 3
        assert len(optimized_model.graph.value_info) == 1
        assert optimized_model.graph.value_info[0].type.tensor_type.elem_type == TensorProto.INT64
        assert len(
            optimized_model.graph.value_info[0].type.tensor_type.shape.dim) == 1
        assert optimized_model.graph.node[0].op_type == 'Constant'
        assert optimized_model.graph.node[1].op_type == 'Tile'
        assert optimized_model.graph.node[2].op_type == 'Conv'
        assert optimized_model.graph.output[0].name == 'B'

    def test_fuse_add_bias_into_conv_use_conv_shape(self):  # type: () -> None
        sub = helper.make_node("Sub", ["M", "N"], ["Y"])
        conv = helper.make_node("Conv", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "A"], ["B"])
        graph = helper.make_graph(
            [sub, conv, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3)),
             helper.make_tensor_value_info(
                 "M", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info(
                 "N", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (1, 16, 1, 1))],
            [helper.make_tensor_value_info(
                "B", TensorProto.FLOAT, (1, 16, 1, 1))],
            value_info=[
                helper.make_tensor_value_info(
                    "Z", TensorProto.FLOAT, (1, 16, 1, 1))
            ],
        )
        optimized_model = self._optimized(graph, ["fuse_add_bias_into_conv"])

        assert len(optimized_model.graph.node) == 3
        assert optimized_model.graph.node[0].op_type == 'Sub'
        assert optimized_model.graph.node[1].op_type == 'Squeeze'
        assert optimized_model.graph.node[2].op_type == 'Conv'
        assert optimized_model.graph.output[0].name == 'B'
        assert optimized_model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
        assert len(
            optimized_model.graph.output[0].type.tensor_type.shape.dim) == 4

    # type: () -> None
    def test_fuse_add_bias_into_conv_use_move_constant(self):
        conv = helper.make_node("Conv", ["X", "Y"], ["Z"])
        constant = helper.make_node("Constant", [], ["A"],
                                    value=helper.make_tensor(
                                        name="bias",
                                        data_type=TensorProto.FLOAT,
                                        dims=(16, 1, 1),
                                        vals=np.random.randn(16).astype(np.float32).tolist()))
        add = helper.make_node("Add", ["Z", "A"], ["B"])
        graph = helper.make_graph(
            [conv, constant, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3))],
            [helper.make_tensor_value_info(
                "B", TensorProto.FLOAT, (1, 16, 1, 1))],
            value_info=[
                helper.make_tensor_value_info(
                    "A", TensorProto.FLOAT, (16, 1, 1)),
            ]
        )
        optimized_model = self._optimized(graph, ["fuse_add_bias_into_conv"])

        assert len(optimized_model.graph.node) == 3
        assert optimized_model.graph.node[0].op_type == 'Constant'
        assert optimized_model.graph.node[1].op_type == 'Squeeze'
        assert optimized_model.graph.node[2].op_type == 'Conv'
        assert optimized_model.graph.output[0].name == 'B'
        assert optimized_model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
        assert len(
            optimized_model.graph.output[0].type.tensor_type.shape.dim) == 4

    # type: () -> None
    def test_fuse_add_bias_into_conv_squeeze_1d_bias_no_fuse(self):
        conv = helper.make_node("Conv", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "A"], ["B"])
        graph = helper.make_graph(
            [conv, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3)),
             helper.make_tensor_value_info(
                 "Y", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (3,))],
            [helper.make_tensor_value_info(
                "B", TensorProto.FLOAT, (1, 16, 1, 3))],
            value_info=[
                helper.make_tensor_value_info(
                    "Z", TensorProto.FLOAT, (1, 16, 1, 1)),
            ]
        )
        optimized_model = self._optimized(graph, ["fuse_add_bias_into_conv"])

        assert len(list(optimized_model.graph.node)) == 2
        assert optimized_model.graph.node[0].op_type == 'Conv'
        assert optimized_model.graph.node[1].op_type == 'Add'

    # type: () -> None
    def test_fuse_add_bias_into_conv_squeeze_3d_bias_no_fuse(self):
        conv = helper.make_node("Conv", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "A"], ["B"])
        graph = helper.make_graph(
            [conv, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3)),
             helper.make_tensor_value_info(
                 "Y", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (16, 3, 3))],
            [helper.make_tensor_value_info(
                "B", TensorProto.FLOAT, (1, 16, 3, 3))],
            value_info=[
                helper.make_tensor_value_info(
                    "Z", TensorProto.FLOAT, (1, 16, 1, 1)),
            ]
        )
        optimized_model = self._optimized(graph, ["fuse_add_bias_into_conv"])

        assert len(list(optimized_model.graph.node)) == 2
        assert optimized_model.graph.node[0].op_type == 'Conv'
        assert optimized_model.graph.node[1].op_type == 'Add'

    # type: () -> None
    def test_fuse_add_bias_into_conv_squeeze_4d_bias_no_fuse(self):
        conv = helper.make_node("Conv", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "A"], ["B"])
        graph = helper.make_graph(
            [conv, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3)),
             helper.make_tensor_value_info(
                 "Y", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (1, 16, 3, 3))],
            [helper.make_tensor_value_info(
                "B", TensorProto.FLOAT, (1, 16, 3, 3))]
        )
        optimized_model = self._optimized(graph, ["fuse_add_bias_into_conv"])

        assert len(list(optimized_model.graph.node)) == 2
        assert optimized_model.graph.node[0].op_type == 'Conv'
        assert optimized_model.graph.node[1].op_type == 'Add'

    def test_fuse_matmul_add_bias_into_gemm(self):  # type: () -> None
        matmul = helper.make_node("MatMul", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "B"], ["A"])
        graph = helper.make_graph(
            [matmul, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (32, 10)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (10, 16)),
             helper.make_tensor_value_info("B", TensorProto.FLOAT, (16,))],
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (32, 16))]
        )
        optimized_model = self._optimized(
            graph, ["fuse_matmul_add_bias_into_gemm"])

        assert len(list(optimized_model.graph.node)) == 1
        assert optimized_model.graph.node[0].op_type == "Gemm"

    def test_fuse_matmul_add_bias_into_gemm_2d_bias(self):  # type: () -> None
        matmul = helper.make_node("MatMul", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "B"], ["A"])
        graph = helper.make_graph(
            [matmul, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (32, 10)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (10, 16)),
             helper.make_tensor_value_info("B", TensorProto.FLOAT, (1, 16))],
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (32, 16))]
        )
        optimized_model = self._optimized(
            graph, ["fuse_matmul_add_bias_into_gemm"])

        assert len(list(optimized_model.graph.node)) == 1
        assert optimized_model.graph.node[0].op_type == "Gemm"

    # type: () -> None
    def test_fuse_matmul_add_bias_into_gemm_2d_bias_same_shape(self):
        matmul = helper.make_node("MatMul", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "B"], ["A"])
        graph = helper.make_graph(
            [matmul, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (32, 10)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (10, 16)),
             helper.make_tensor_value_info("B", TensorProto.FLOAT, (32, 16))],
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (32, 16))]
        )
        optimized_model = self._optimized(
            graph, ["fuse_matmul_add_bias_into_gemm"])

        assert len(list(optimized_model.graph.node)) == 1
        assert optimized_model.graph.node[0].op_type == "Gemm"

    # type: () -> None
    def test_fuse_matmul_add_bias_into_gemm_2d_bias_bcast_no_fuse(self):
        matmul = helper.make_node("MatMul", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "B"], ["A"])
        graph = helper.make_graph(
            [matmul, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 10)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (10, 16)),
             helper.make_tensor_value_info("B", TensorProto.FLOAT, (16, 16))],
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (16, 16))]
        )
        optimized_model = self._optimized(
            graph, ["fuse_matmul_add_bias_into_gemm"])

        assert optimized_model.graph == graph

    # type: () -> None
    def test_fuse_matmul_add_bias_into_gemm_3d_matmul_no_fuse(self):
        matmul = helper.make_node("MatMul", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "B"], ["A"])
        graph = helper.make_graph(
            [matmul, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 4, 3)),
             helper.make_tensor_value_info("B", TensorProto.FLOAT, (3, 3))],
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3, 3))]
        )
        optimized_model = self._optimized(
            graph, ["fuse_matmul_add_bias_into_gemm"])

        assert optimized_model.graph == graph

    # type: () -> None
    def test_fuse_matmul_add_bias_into_gemm_3d_bias_no_fuse(self):
        matmul = helper.make_node("MatMul", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "B"], ["A"])
        graph = helper.make_graph(
            [matmul, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (32, 10)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (10, 16)),
             helper.make_tensor_value_info("B", TensorProto.FLOAT, (4, 1, 16))],
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (32, 16))]
        )
        # 3d bias for 2d matmul is not legal. So disable onnxruntime checking
        optimized_model = self._optimized(
            graph, ["fuse_matmul_add_bias_into_gemm"], compare_result=False)

        assert optimized_model.graph == graph

    # type: () -> None
    def test_fuse_matmul_add_bias_into_gemm_multiple_use_no_fuse(self):
        matmul = helper.make_node("MatMul", ["X", "Y"], ["Z"])
        identity = helper.make_node("Identity", ["Z"], ["A1"])
        add = helper.make_node("Add", ["Z", "B"], ["A2"])
        graph = helper.make_graph(
            [matmul, add, identity],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (32, 10)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (10, 16)),
             helper.make_tensor_value_info("B", TensorProto.FLOAT, (1, 16))],
            [helper.make_tensor_value_info("A1", TensorProto.FLOAT, (32, 16)),
             helper.make_tensor_value_info("A2", TensorProto.FLOAT, (32, 16))]
        )
        optimized_model = self._optimized(
            graph, ["fuse_matmul_add_bias_into_gemm"])

        assert optimized_model.graph == graph

    # type: () -> None
    def test_fuse_pad_into_conv_no_optional_value_opset10(self):
        pad = helper.make_node(
            "Pad",
            ["X"],
            ["P"],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 1, 1]
        )
        conv = helper.make_node("Conv", ["P", "Y"], ["Z"])
        graph = helper.make_graph(
            [pad, conv],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 2, 2)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3))],
            [helper.make_tensor_value_info(
                "Z", TensorProto.FLOAT, (1, 16, 1, 1))]
        )
        optimized_model = self._optimized(
            graph, ["fuse_pad_into_conv"], False, opset_imports=[helper.make_opsetid("", 10)])

        assert len(list(optimized_model.graph.node)) == 1
        assert optimized_model.graph.node[0].op_type == "Conv"
        assert optimized_model.graph.node[0].attribute[0].name == "pads"
        assert list(optimized_model.graph.node[0].attribute[0].ints) == [
            0, 0, 1, 1]

    def test_fuse_pad_into_conv_no_optional_value(self):  # type: () -> None
        pad = helper.make_node(
            "Pad",
            ["X", "Pads"],
            ["P"],
            mode="constant"
        )
        conv = helper.make_node("Conv", ["P", "Y"], ["Z"])
        graph = helper.make_graph(
            [pad, conv],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 2, 2)),
             helper.make_tensor_value_info("Pads", TensorProto.INT64, (8,)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3))],
            [helper.make_tensor_value_info(
                "Z", TensorProto.FLOAT, (1, 16, 1, 1))],
            [helper.make_tensor("Pads", TensorProto.INT64,
                                dims=(8,),
                                vals=np.array([0, 0, 0, 0, 0, 0, 1, 1]).astype(
                                    np.int64).tobytes(),
                                raw=True)])
        optimized_model = self._optimized(graph, ["fuse_pad_into_conv"])

        assert len(list(optimized_model.graph.node)) == 1
        assert optimized_model.graph.node[0].op_type == "Conv"
        assert optimized_model.graph.node[0].attribute[0].name == "pads"
        assert list(optimized_model.graph.node[0].attribute[0].ints) == [
            0, 0, 1, 1]

    def test_fuse_pad_into_conv_with_optional_value(self):  # type: () -> None
        pad = helper.make_node(
            "Pad",
            ["X", "Pads", "Constant_value"],
            ["P"],
            mode="constant"
        )
        conv = helper.make_node("Conv", ["P", "Y"], ["Z"])
        graph = helper.make_graph(
            [pad, conv],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 2, 2)),
             helper.make_tensor_value_info("Pads", TensorProto.INT64, (8,)),
             helper.make_tensor_value_info(
                 "Constant_value", TensorProto.FLOAT, ()),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3))],
            [helper.make_tensor_value_info(
                "Z", TensorProto.FLOAT, (1, 16, 1, 1))],
            [helper.make_tensor("Pads", TensorProto.INT64,
                                dims=(8,),
                                vals=np.array([0, 0, 0, 0, 0, 0, 1, 1]).astype(
                                    np.int64).tobytes(),
                                raw=True),
             helper.make_tensor("Constant_value", TensorProto.FLOAT,
                                dims=(),
                                vals=np.array([0]).astype(np.float32).tobytes(),
                                raw=True)])
        optimized_model = self._optimized(graph, ["fuse_pad_into_conv"])

        assert len(list(optimized_model.graph.node)) == 1
        assert optimized_model.graph.node[0].op_type == "Conv"
        assert optimized_model.graph.node[0].attribute[0].name == "pads"
        assert list(optimized_model.graph.node[0].attribute[0].ints) == [
            0, 0, 1, 1]

    # type: () -> None
    def test_fuse_pad_into_conv_with_nonzero_optional_value(self):
        pad = helper.make_node(
            "Pad",
            ["X", "Pads", "Constant_value"],
            ["P"],
            mode="constant"
        )
        conv = helper.make_node("Conv", ["P", "Y"], ["Z"])
        graph = helper.make_graph(
            [pad, conv],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 2, 2)),
             helper.make_tensor_value_info("Pads", TensorProto.INT64, (8,)),
             helper.make_tensor_value_info(
                 "Constant_value", TensorProto.FLOAT, ()),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3))],
            [helper.make_tensor_value_info(
                "Z", TensorProto.FLOAT, (1, 16, 1, 1))],
            [helper.make_tensor("Pads", TensorProto.INT64,
                                dims=(8,),
                                vals=np.array([0, 0, 0, 0, 0, 0, 1, 1]).astype(
                                    np.int64).tobytes(),
                                raw=True),
             helper.make_tensor("Constant_value", TensorProto.FLOAT,
                                dims=(),
                                # non-zero Constant_value -> so no pad
                                vals=np.array([25]).astype(
                                    np.float32).tobytes(),
                                raw=True)])
        optimized_model = self._optimized(graph, ["fuse_pad_into_conv"])

        assert optimized_model.graph == graph

    def test_fuse_pad_into_conv_1d_opset10(self):  # type: () -> None
        pad = helper.make_node(
            "Pad",
            ["X"],
            ["P"],
            mode="constant",
            pads=[0, 0, 1, 0, 0, 1]
        )
        conv = helper.make_node("Conv", ["P", "Y"], ["Z"])
        graph = helper.make_graph(
            [pad, conv],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 30)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 32))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 16, 1))]
        )
        optimized_model = self._optimized(
            graph, ["fuse_pad_into_conv"], False, opset_imports=[helper.make_opsetid("", 10)])

        assert len(list(optimized_model.graph.node)) == 1
        assert optimized_model.graph.node[0].op_type == "Conv"
        assert optimized_model.graph.node[0].attribute[0].name == "pads"
        assert list(optimized_model.graph.node[0].attribute[0].ints) == [1, 1]

    def test_fuse_pad_into_conv_1d(self):  # type: () -> None
        pad = helper.make_node(
            "Pad",
            ["X", "Pads"],
            ["P"],
            mode="constant"
        )
        conv = helper.make_node("Conv", ["P", "Y"], ["Z"])
        graph = helper.make_graph(
            [pad, conv],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 30)),
             helper.make_tensor_value_info("Pads", TensorProto.INT64, (6,)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 32))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 16, 1))],
            [helper.make_tensor("Pads", TensorProto.INT64,
                                dims=(6,),
                                vals=np.array([0, 0, 1, 0, 0, 1]).astype(
                                    np.int64).tobytes(),
                                raw=True)])
        optimized_model = self._optimized(graph, ["fuse_pad_into_conv"])

        assert len(list(optimized_model.graph.node)) == 1
        assert optimized_model.graph.node[0].op_type == "Conv"
        assert optimized_model.graph.node[0].attribute[0].name == "pads"
        assert list(optimized_model.graph.node[0].attribute[0].ints) == [1, 1]

    # type: () -> None
    def test_fuse_pad_into_conv_existing_conv_pad_opset10(self):
        pad = helper.make_node(
            "Pad",
            ["X"],
            ["P"],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 1, 1]
        )
        conv = helper.make_node(
            "Conv",
            ["P", "Y"],
            ["Z"],
            pads=[1, 1, 0, 0]
        )
        graph = helper.make_graph(
            [pad, conv],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 2, 2)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 4, 4))],
            [helper.make_tensor_value_info(
                "Z", TensorProto.FLOAT, (1, 16, 1, 1))]
        )
        optimized_model = self._optimized(
            graph, ["fuse_pad_into_conv"], False, opset_imports=[helper.make_opsetid("", 10)])

        assert len(list(optimized_model.graph.node)) == 1
        assert optimized_model.graph.node[0].op_type == "Conv"
        assert optimized_model.graph.node[0].attribute[0].name == "pads"
        assert list(optimized_model.graph.node[0].attribute[0].ints) == [
            1, 1, 1, 1]

    def test_fuse_pad_into_conv_existing_conv_pad(self):  # type: () -> None
        pad = helper.make_node(
            "Pad",
            ["X", "Pads"],
            ["P"],
            mode="constant"
        )
        conv = helper.make_node(
            "Conv",
            ["P", "Y"],
            ["Z"],
            pads=[1, 1, 0, 0]
        )
        graph = helper.make_graph(
            [pad, conv],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 2, 2)),
             helper.make_tensor_value_info("Pads", TensorProto.INT64, (8,)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 4, 4))],
            [helper.make_tensor_value_info(
                "Z", TensorProto.FLOAT, (1, 16, 1, 1))],
            [helper.make_tensor("Pads", TensorProto.INT64,
                                dims=(8,),
                                vals=np.array([0, 0, 0, 0, 0, 0, 1, 1]).astype(
                                    np.int64).tobytes(),
                                raw=True)])
        optimized_model = self._optimized(graph, ["fuse_pad_into_conv"])

        assert len(list(optimized_model.graph.node)) == 1
        assert optimized_model.graph.node[0].op_type == "Conv"
        assert optimized_model.graph.node[0].attribute[0].name == "pads"
        assert list(optimized_model.graph.node[0].attribute[0].ints) == [
            1, 1, 1, 1]

    # type: () -> None
    def test_fuse_pad_into_conv_pad_feature_no_fuse_opset10(self):
        pad = helper.make_node(
            "Pad",
            ["X"],
            ["P"],
            mode="constant",
            pads=[0, 1, 0, 0, 0, 0, 0, 0]
        )
        conv = helper.make_node("Conv", ["P", "Y"], ["Z"])
        graph = helper.make_graph(
            [pad, conv],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 4, 3, 3)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3))],
            [helper.make_tensor_value_info(
                "Z", TensorProto.FLOAT, (1, 16, 1, 1))]
        )
        optimized_model = self._optimized(
            graph, ["fuse_pad_into_conv"], False, opset_imports=[helper.make_opsetid("", 10)])

        assert optimized_model.graph == graph

    def test_fuse_pad_into_conv_pad_feature_no_fuse(self):  # type: () -> None
        pad = helper.make_node(
            "Pad",
            ["X", "Pads"],
            ["P"],
            mode="constant"
        )
        conv = helper.make_node("Conv", ["P", "Y"], ["Z"])
        graph = helper.make_graph(
            [pad, conv],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 4, 3, 3)),
             helper.make_tensor_value_info("Pads", TensorProto.INT64, (8,)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3))],
            [helper.make_tensor_value_info(
                "Z", TensorProto.FLOAT, (1, 16, 1, 1))],
            [helper.make_tensor("Pads", TensorProto.INT64,
                                dims=(8,),
                                vals=np.array([0, 1, 0, 0, 0, 0, 0, 0]).astype(
                                    np.int64).tobytes(),
                                raw=True)])
        optimized_model = self._optimized(graph, ["fuse_pad_into_conv"])

        assert optimized_model.graph == graph

    # type: () -> None
    def test_fuse_pad_into_conv_negative_pad_no_fuse_opset10(self):
        pad = helper.make_node(
            "Pad",
            ["X"],
            ["P"],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, -1, -1]
        )
        conv = helper.make_node("Conv", ["P", "Y"], ["Z"])
        graph = helper.make_graph(
            [pad, conv],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 4, 4)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3))],
            [helper.make_tensor_value_info(
                "Z", TensorProto.FLOAT, (1, 16, 1, 1))]
        )
        optimized_model = self._optimized(
            graph, ["fuse_pad_into_conv"], False, opset_imports=[helper.make_opsetid("", 10)])

        assert optimized_model.graph == graph

    def test_fuse_pad_into_conv_negative_pad_no_fuse(self):  # type: () -> None
        pad = helper.make_node(
            "Pad",
            ["X", "Pads"],
            ["P"],
            mode="constant"
        )
        conv = helper.make_node("Conv", ["P", "Y"], ["Z"])
        graph = helper.make_graph(
            [pad, conv],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 4, 4)),
             helper.make_tensor_value_info("Pads", TensorProto.INT64, (8,)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3))],
            [helper.make_tensor_value_info(
                "Z", TensorProto.FLOAT, (1, 16, 1, 1))],
            [helper.make_tensor("Pads", TensorProto.INT64,
                                dims=(8,),
                                vals=np.array(
                                    [0, 0, 0, 0, 0, 0, -1, -1]).astype(np.int64).tobytes(),
                                raw=True)])
        optimized_model = self._optimized(graph, ["fuse_pad_into_conv"])

        assert optimized_model.graph == graph

    # type: () -> None
    def test_fuse_pad_into_conv_reflection_pad_no_fuse_opset10(self):
        pad = helper.make_node(
            "Pad",
            ["X"],
            ["P"],
            mode="reflect",
            pads=[0, 0, 0, 0, 0, 0, 1, 1]
        )
        conv = helper.make_node("Conv", ["P", "Y"], ["Z"])
        graph = helper.make_graph(
            [pad, conv],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 2, 2)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3))],
            [helper.make_tensor_value_info(
                "Z", TensorProto.FLOAT, (1, 16, 1, 1))]
        )
        optimized_model = self._optimized(
            graph, ["fuse_pad_into_conv"], False, opset_imports=[helper.make_opsetid("", 10)])

        assert optimized_model.graph == graph

    # type: () -> None
    def test_fuse_pad_into_conv_reflection_pad_no_fuse(self):
        pad = helper.make_node(
            "Pad",
            ["X", "Pads"],
            ["P"],
            mode="reflect"
        )
        conv = helper.make_node("Conv", ["P", "Y"], ["Z"])
        graph = helper.make_graph(
            [pad, conv],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 2, 2)),
             helper.make_tensor_value_info("Pads", TensorProto.INT64, (8,)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3))],
            [helper.make_tensor_value_info(
                "Z", TensorProto.FLOAT, (1, 16, 1, 1))],
            [helper.make_tensor("Pads", TensorProto.INT64,
                                dims=(8,),
                                vals=np.array([0, 0, 0, 0, 0, 0, 1, 1]).astype(
                                    np.int64).tobytes(),
                                raw=True)])
        optimized_model = self._optimized(graph, ["fuse_pad_into_conv"])

        assert optimized_model.graph == graph

    def test_fuse_consecutive_squeezes(self):  # type: () -> None
        nodes = [helper.make_node("Squeeze", ["X", "X_axes"], ["Y"]),
                 helper.make_node("Squeeze", ["Y", "Y_axes"], ["Z"])]
        nodes.extend(self._make_fake_loop_op(
            [helper.make_node("Squeeze", ["_X", "X_axes"], ["_Y"]),
             helper.make_node("Squeeze", ["_Y", "Y_axes"], ["_Z2"])],
            [(TensorProto.FLOAT, (1, 1, 2, 3, 1, 1, 1, 1, 8, 9), "X")],
            [(TensorProto.FLOAT, (2, 3, 1, 8, 9), "Z2")]))
        initializers = [
            helper.make_tensor(name, TensorProto.INT64,
                               npa.shape, npa.tobytes(), raw=True)
            for name, npa in [('X_axes', np.array([0, 4, 5], dtype=np.int64)),
                              ('Y_axes', np.array([0, 3], dtype=np.int64))]
        ]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info(
                "X", TensorProto.FLOAT, (1, 1, 2, 3, 1, 1, 1, 1, 8, 9)),
                helper.make_tensor_value_info("X_axes", TensorProto.INT64, [3]),
                helper.make_tensor_value_info("Y_axes", TensorProto.INT64, [2])],
            [helper.make_tensor_value_info(
                "Z", TensorProto.FLOAT, (2, 3, 1, 8, 9))],
            initializer=initializers)
        optimized_model = self._optimized(graph, ["fuse_consecutive_squeezes"])

        # Squeeze, Constant (trip count), Constant (cond), Loop
        assert optimized_model.graph.node[0].op_type == "Squeeze"
        for init in optimized_model.graph.initializer:
            if init.name == optimized_model.graph.node[0].input[1]:
                assert list(to_array(init)) == [0, 1, 4, 5, 6]
        assert len(list(optimized_model.graph.node)) == 4

    def test_fuse_consecutive_squeezes_default(self):  # type: () -> None
        squeeze1 = helper.make_node("Squeeze", ["X", "X_axes"], ["Y"])
        squeeze2 = helper.make_node("Squeeze", ["Y", "Y_axes"], ["Z"])
        squeeze3 = helper.make_node("Squeeze", ["Z", "Z_axes"], ["A"])
        nodes = [squeeze1, squeeze2, squeeze3]
        initializers = [
            helper.make_tensor(name, TensorProto.INT64,
                               npa.shape, npa.tobytes(), raw=True)
            for name, npa in [('X_axes', np.array([0, 4, 5], dtype=np.int64)),
                              ('Y_axes', np.array([0, 3], dtype=np.int64)),
                              ('Z_axes', np.array([2], dtype=np.int64))]
        ]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info(
                "X", TensorProto.FLOAT, (1, 1, 2, 3, 1, 1, 1, 1, 8, 9)),
                helper.make_tensor_value_info("X_axes", TensorProto.INT64, [3]),
                helper.make_tensor_value_info("Y_axes", TensorProto.INT64, [2]),
                helper.make_tensor_value_info("Z_axes", TensorProto.INT64, [1])],
            [helper.make_tensor_value_info(
                "A", TensorProto.FLOAT, (2, 3, 8, 9))],
            initializer=initializers)
        optimized_model = self._optimized(graph, ["fuse_consecutive_squeezes"])

        assert optimized_model.graph.node[0].op_type == "Squeeze"
        for init in optimized_model.graph.initializer:
            if init.name == optimized_model.graph.node[0].input[1]:
                assert list(to_array(init)) == [0, 1, 4, 5, 6, 7]
        assert len(list(optimized_model.graph.node)) == 1

    def test_fuse_consecutive_squeezes_random(self):  # type: () -> None
        x_shape = [1, 1, 1, 3, 4, 1, 6, 1, 1, 9]
        s1_one_indices = [i for i, a in enumerate(x_shape) if a == 1]
        s1_axes = np.random.choice(s1_one_indices,
                                   size=np.random.randint(
                                       low=1, high=len(s1_one_indices) - 1),
                                   replace=False).astype(np.int64)
        s2_x_shape = [a for i, a in enumerate(x_shape) if i not in s1_axes]
        s2_one_indices = [i for i, a in enumerate(s2_x_shape) if a == 1]
        s2_axes = np.array(s2_one_indices).astype(np.int64)

        squeeze1 = helper.make_node("Squeeze", ["X", "X_axes"], ["Y"])
        squeeze2 = helper.make_node("Squeeze", ["Y", "Y_axes"], ["Z"])
        initializers = [
            helper.make_tensor(name, TensorProto.INT64,
                               npa.shape, npa.tobytes(), raw=True)
            for name, npa in [('X_axes', s1_axes),
                              ('Y_axes', s2_axes)]
        ]
        nodes = [squeeze1, squeeze2]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
             helper.make_tensor_value_info(
                 "X_axes", TensorProto.INT64, s1_axes.shape),
             helper.make_tensor_value_info("Y_axes", TensorProto.INT64, s2_axes.shape)],
            [helper.make_tensor_value_info(
                "Z", TensorProto.FLOAT, (3, 4, 6, 9))],
            initializer=initializers
        )
        optimized_model = self._optimized(graph, ["fuse_consecutive_squeezes"])

        assert optimized_model.graph.node[0].op_type == "Squeeze"
        for init in optimized_model.graph.initializer:
            if init.name == optimized_model.graph.node[0].input[1]:
                assert list(to_array(init)) == [0, 1, 2, 5, 7, 8]
        assert len(list(optimized_model.graph.node)) == 1

    def test_fuse_consecutive_squeezes_multi_uses(self):  # type: () -> None
        squeeze1 = helper.make_node("Squeeze", ["X", "X_axes"], ["Y"])
        add = helper.make_node("Add", ["Y", "A"], ["Z2"])
        squeeze2 = helper.make_node("Squeeze", ["Y", "Y_axes"], ["Z"])
        initializers = [
            helper.make_tensor(name, TensorProto.INT64,
                               npa.shape, npa.tobytes(), raw=True)
            for name, npa in [('X_axes', np.array([0, 4, 5], dtype=np.int64)),
                              ('Y_axes', np.array([0, 3], dtype=np.int64)), ]
        ]
        graph = helper.make_graph(
            [squeeze1, add, squeeze2],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 1, 2, 3, 1, 1, 1, 1, 8, 9)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (1,)),
             helper.make_tensor_value_info("X_axes", TensorProto.INT64, [3]),
             helper.make_tensor_value_info("Y_axes", TensorProto.INT64, [2]),
             ],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (2, 3, 1, 8, 9)),
             helper.make_tensor_value_info("Z2", TensorProto.FLOAT, (1, 2, 3, 1, 1, 8, 9))],
            initializer=initializers
        )
        optimized_model = self._optimized(graph, ["fuse_consecutive_squeezes"])

        assert optimized_model.graph.node[0].op_type == "Squeeze"
        assert optimized_model.graph.node[2].op_type == "Squeeze"
        assert optimized_model.graph.node[2].input[0] == "X"
        assert len(list(optimized_model.graph.node)) == 3
        for init in optimized_model.graph.initializer:
            if init.name == optimized_model.graph.node[0].input[1]:
                assert list(to_array(init)) == [
                    0, 4, 5]
            if init.name == optimized_model.graph.node[2].input[1]:
                assert list(to_array(init)) == [
                    0, 1, 4, 5, 6]

    def test_fuse_consecutive_softmax_log_axis(self):  # type: () -> None
        for axis in range(3):
            softmax = helper.make_node("Softmax", ["X"], ["Y"], axis=axis)
            log = helper.make_node("Log", ["Y"], ["Z"])
            graph = helper.make_graph(
                [softmax, log],
                "test",
                [helper.make_tensor_value_info(
                    "X", TensorProto.FLOAT, (5, 7, 11))],
                [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (5, 7, 11))])
            optimized_model = self._optimized(
                graph, ["fuse_consecutive_log_softmax"])

            assert optimized_model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
            assert len(optimized_model.graph.output) == 1
            assert len(optimized_model.graph.node) == 1
            assert optimized_model.graph.node[0].op_type == "LogSoftmax"
            assert optimized_model.graph.node[0].attribute[0].name == "axis"
            assert optimized_model.graph.node[0].attribute[0].i == axis

    def test_fuse_consecutive_softmax_log_side_effect(self):  # type: () -> None
        softmax = helper.make_node("Softmax", ["X"], ["Y"], axis=2)
        log = helper.make_node("Log", ["Y"], ["Z"])
        graph = helper.make_graph(
            [softmax, log],
            "test",
            [helper.make_tensor_value_info(
                "X", TensorProto.FLOAT, (5, 7, 11))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (5, 7, 11)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5, 7, 11))])
        optimized_model = self._optimized(
            graph, ["fuse_consecutive_log_softmax"])

        assert graph == optimized_model.graph

    # type: () -> None
    def test_fuse_consecutive_softmax_log_multiple_out(self):
        softmax = helper.make_node("Softmax", ["X"], ["Y"], axis=2)
        log = helper.make_node("Log", ["Y"], ["Z"])
        exp = helper.make_node("Exp", ["Z"], ["Z1"])
        graph = helper.make_graph(
            [softmax, log, exp],
            "test",
            [helper.make_tensor_value_info(
                "X", TensorProto.FLOAT, (5, 7, 11))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (5, 7, 11)),
             helper.make_tensor_value_info("Z1", TensorProto.FLOAT, (5, 7, 11))])
        optimized_model = self._optimized(
            graph, ["fuse_consecutive_log_softmax"])

        assert len(optimized_model.graph.output) == 2
        assert len(optimized_model.graph.node) == 2
        assert optimized_model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
        assert optimized_model.graph.output[1].type.tensor_type.elem_type == TensorProto.FLOAT
        assert optimized_model.graph.node[0].op_type == "LogSoftmax"
        assert optimized_model.graph.node[0].attribute[0].name == "axis"
        assert optimized_model.graph.node[0].attribute[0].i == 2
        assert optimized_model.graph.node[1].op_type == "Exp"

    def test_preserve_value_info(self):  # type: () -> None
        trans1 = helper.make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])
        trans2 = helper.make_node("Transpose", ["Y"], ["Z"], perm=[2, 0, 1])
        trans3 = helper.make_node("Transpose", ["Z"], ["A"], perm=[2, 0, 1])
        graph = helper.make_graph(
            [trans1, trans2, trans3],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4))],
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 4, 3))])
        vi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, (3, 2, 4))
        graph.value_info.extend([vi])
        optimized_model = self._optimized(graph, ["nop"])
        assert list(optimized_model.graph.value_info) == [vi]
        assert len(list(optimized_model.graph.node)) == 3

    def test_split(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['X'],
            value=onnx.helper.make_tensor(
                name='X',
                data_type=TensorProto.FLOAT,
                dims=[1],
                vals=[5],
            ),
        )
        graph = helper.make_graph(
            [node],
            'test-optimize-split',
            [],
            [helper.make_tensor_value_info('X', TensorProto.FLOAT, (1,))])

        init_model = self._optimized(graph, ['split_init'])
        self.assertEqual(len(init_model.graph.node), 1)
        self.assertEqual(len(init_model.graph.output), 1)
        self.assertEqual(init_model.graph.node[0].op_type, 'Constant')

        predict_model = self._optimized(graph, ['split_predict'])
        self.assertEqual(len(predict_model.graph.node), 0)
        self.assertEqual(len(predict_model.graph.input), 1)
        self.assertEqual(predict_model.graph.input[0].name, 'X')

    def test_lift_lex_loop(self):  # type: () -> None
        nodes = [helper.make_node("Identity", ["X"], ["Y"])]
        # 'lift_lexical_references' is legacy code and I don't know how it works.
        # More error occurs if I make this loop op legal.
        # So don't check legality here
        nodes.extend(self._make_fake_loop_op(
            [helper.make_node("Identity", ["X"], ["_Y2"]),
             helper.make_node("Identity", ["Y"], ["_Y3"])],
            [],
            [(TensorProto.FLOAT, (5,), "Y2"),
             (TensorProto.FLOAT, (5,), "Y3")],
            check_legality=False))
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,)),
             helper.make_tensor_value_info("Y2", TensorProto.FLOAT, (5,))])
        # "lift_lexical_references" pass produces a graph that does not conform to
        # the ONNX spec. Disable checking.
        optimized_model = self._optimized(
            graph, ["lift_lexical_references"], compare_result=False)
        assert len(optimized_model.graph.node) == 4
        # body_graph, __control_inputs
        assert len(optimized_model.graph.node[3].attribute) == 2
        assert optimized_model.graph.node[3].attribute[1].name == "__control_inputs"
        assert optimized_model.graph.node[3].attribute[1].strings[0] == b"X"
        assert optimized_model.graph.node[3].attribute[1].strings[1] == b"Y"

    def test_lift_lex_if(self):  # type: () -> None
        nodes = [helper.make_node("Identity", ["X"], ["Y"])]
        nodes.extend(self._make_fake_if_op(
            [helper.make_node("Identity", ["X"], ["_Y2"]),
             helper.make_node("Identity", ["Y"], ["_Y3"])],
            [helper.make_node("Identity", ["X"], ["_Y2"]),
             helper.make_node("Identity", ["X"], ["_Y3"])],
            [(TensorProto.FLOAT, (5,), "Y2"),
             (TensorProto.FLOAT, (5,), "Y3")]))
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,)),
             helper.make_tensor_value_info("Y2", TensorProto.FLOAT, (5,))])
        # "If" node now diverges from ONNX schema. Disable checking.
        optimized_model = self._optimized(
            graph, ["lift_lexical_references"], compare_result=False)

        # Identity, Constant (condition), If
        assert len(optimized_model.graph.node) == 3
        # else_branch, then_branch, __control_inputs
        assert len(optimized_model.graph.node[2].attribute) == 3
        assert optimized_model.graph.node[2].attribute[2].name == "__control_inputs"
        assert optimized_model.graph.node[2].attribute[2].strings[0] == b"X"
        assert optimized_model.graph.node[2].attribute[2].strings[1] == b"Y"

    def test_fuse_bn_into_conv_simple(self):  # type: () -> None
        for (tensor_type, np_type) in [(TensorProto.FLOAT, np.float32)]:
            conv = helper.make_node("Conv", ["X", "W", "B"], ["Y"])
            bn = helper.make_node("BatchNormalization", [
                                  "Y", "scale", "b", "mean", "var"], ["Z"])

            W = np.random.randn(3, 2, 5, 5).astype(np_type) + 2
            B = np.random.randn(3,).astype(np_type) + 2
            scale = np.random.randn(3,).astype(np_type) + 2
            b = np.random.randn(3,).astype(np_type) + 2
            mean = np.random.randn(3,).astype(np_type) + 2
            var = np.abs(np.random.randn(3,).astype(np_type)) + 2

            initializers = [
                helper.make_tensor(name, tensor_type,
                                   npa.shape, npa.tobytes(), raw=True)
                for name, npa in [('W', W), ('B', B), ('scale', scale), ('b', b), ('mean', mean), ('var', var)]
            ]
            graph = helper.make_graph(
                [conv, bn],
                "test",
                [helper.make_tensor_value_info(
                    "X", tensor_type, (5, 2, 28, 28))],
                [helper.make_tensor_value_info(
                    "Z", tensor_type, (5, 3, 24, 24))],
                initializer=initializers,
                value_info=[
                    helper.make_tensor_value_info(
                        "Y", tensor_type, (5, 3, 24, 24))
                ]
            )
            optimized_model = self._optimized(graph, ["fuse_bn_into_conv"])

            self.assertEqual(len(optimized_model.graph.node), 1)
            self.assertEqual(optimized_model.graph.node[0].op_type, 'Conv')
            self.assertEqual(len(optimized_model.graph.initializer), 2)
            new_W = numpy_helper.to_array(optimized_model.graph.initializer[0])
            new_b = numpy_helper.to_array(optimized_model.graph.initializer[1])

            f = scale / np.sqrt(var + 1e-5)
            np.testing.assert_almost_equal((B - mean) * f + b, new_b)
            np.testing.assert_almost_equal(
                W * f[:, np.newaxis, np.newaxis, np.newaxis], new_W)

    def _internal_test_deadend_elimination(self, fixed):  # type: (bool) -> None
        softmax = helper.make_node("Softmax", ["X"], ["Y"], axis=2)
        log = helper.make_node("Log", ["Y"], ["Z"])
        exp = helper.make_node("Exp", ["Z"], ["Z1"])
        exp1 = helper.make_node("Log", ["Z"], ["Z2"])
        exp2 = helper.make_node("Sqrt", ["Z1"], ["Z3"])
        graph = helper.make_graph(
            [softmax, log, exp, exp1, exp2],
            "test",
            [helper.make_tensor_value_info(
                "X", TensorProto.FLOAT, (5, 7, 11))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (5, 7, 11))])
        optimized_model = self._optimized(
            graph, ["eliminate_deadend"], fixed)
        assert len(optimized_model.graph.output) == 1
        assert len(optimized_model.graph.node) == 2
        assert optimized_model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
        assert optimized_model.graph.node[0].op_type == "Softmax"
        assert optimized_model.graph.node[0].attribute[0].name == "axis"
        assert optimized_model.graph.node[0].attribute[0].i == 2
        assert optimized_model.graph.node[1].op_type == "Log"

    def test_deadend_elimination_simple(self):  # type: () -> None
        self._internal_test_deadend_elimination(False)

    def test_deadend_elimination_simple_fixed(self):  # type: () -> None
        self._internal_test_deadend_elimination(True)

    def _get_argmax_output_shape(self, input_shape, axis, keepdims):
        assert keepdims
        output_shape = list(input_shape[:])
        output_shape[axis] = 1
        output_shape = tuple(output_shape)
        return output_shape

    # type: () -> None

    def test_eliminate_nop_monotone_argmax_basic_no_node_axis(self):
        input_shape = (5, 7, 11)
        for node_name in ["Exp"]:
            for axis in range(3):
                node = helper.make_node(node_name, ["X"], ["Y"])
                argmax = helper.make_node("ArgMax", ["Y"], ["Z"], axis=axis)
                output_shape = self._get_argmax_output_shape(
                    input_shape, axis, True)
                graph = helper.make_graph(
                    [node, argmax],
                    "test",
                    [helper.make_tensor_value_info(
                        "X", TensorProto.FLOAT, input_shape)],
                    [helper.make_tensor_value_info("Z", TensorProto.INT64, output_shape)])
                optimized_model = self._optimized(
                    graph, ["eliminate_nop_monotone_argmax"])
                assert len(optimized_model.graph.output) == 1
                assert len(optimized_model.graph.node) == 1
                assert optimized_model.graph.output[0].type.tensor_type.elem_type == TensorProto.INT64
                assert optimized_model.graph.node[0].op_type == "ArgMax"
                assert optimized_model.graph.node[0].attribute[0].name == "axis"
                assert optimized_model.graph.node[0].attribute[0].i == axis

    # type: () -> None
    def test_eliminate_nop_monotone_argmax_basic_with_node_axis(self):
        input_shape = (5, 7, 11)
        for node_name in ["Softmax", "LogSoftmax"]:
            for axis_n in range(3):
                for axis_max in range(3):
                    node = helper.make_node(
                        node_name, ["X"], ["Y"], axis=axis_n)
                    argmax = helper.make_node(
                        "ArgMax", ["Y"], ["Z"], axis=axis_max)
                    output_shape = self._get_argmax_output_shape(
                        input_shape, axis_max, True)
                    graph = helper.make_graph(
                        [node, argmax],
                        "test",
                        [helper.make_tensor_value_info(
                            "X", TensorProto.FLOAT, input_shape)],
                        [helper.make_tensor_value_info("Z", TensorProto.INT64, output_shape)])
                    optimized_model = self._optimized(
                        graph, ["eliminate_nop_monotone_argmax"])
                    if axis_max == axis_n:
                        assert len(optimized_model.graph.output) == 1
                        assert len(optimized_model.graph.node) == 1
                        assert optimized_model.graph.output[0].type.tensor_type.elem_type == TensorProto.INT64
                        assert optimized_model.graph.node[0].op_type == "ArgMax"
                        assert optimized_model.graph.node[0].attribute[0].name == "axis"
                        assert optimized_model.graph.node[0].attribute[0].i == axis_max
                    else:
                        assert optimized_model.graph == graph

    # type: () -> None
    def test_eliminate_nop_monotone_argmax_multiple_out(self):
        input_shape = (5, 7, 11)
        for node_name in ["Exp"]:
            for axis in range(3):
                node = helper.make_node(node_name, ["X"], ["Y"])
                node2 = helper.make_node(node_name, ["Y"], ["Z1"])
                argmax = helper.make_node("ArgMax", ["Y"], ["Z"], axis=axis)
                argmax_output_shape = self._get_argmax_output_shape(
                    input_shape, axis, True)
                graph = helper.make_graph(
                    [node, node2, argmax],
                    "test",
                    [helper.make_tensor_value_info(
                        "X", TensorProto.FLOAT, input_shape)],
                    [helper.make_tensor_value_info("Z", TensorProto.INT64, argmax_output_shape),
                     helper.make_tensor_value_info("Z1", TensorProto.FLOAT, input_shape)])
                optimized_model = self._optimized(
                    graph, ["eliminate_nop_monotone_argmax"])
                assert optimized_model.graph == graph

    # type: () -> None
    def test_eliminate_nop_monotone_argmax_consecutive(self):
        # type: (GraphProto, ModelProto, bool, int) -> None
        input_shape = (5, 7, 11)

        def _assertion(graph, optimized_model, axis_aligned, true_axis):
            if axis_aligned:
                assert len(optimized_model.graph.output) == 1
                assert len(optimized_model.graph.node) == 1
                assert optimized_model.graph.output[0].type.tensor_type.elem_type == TensorProto.INT64
                assert optimized_model.graph.node[0].op_type == "ArgMax"
                assert optimized_model.graph.node[0].attribute[0].name == "axis"
                assert optimized_model.graph.node[0].attribute[0].i == true_axis
            else:
                assert optimized_model.graph == graph
        # no axis X no axis test
        for node_name_0 in ["Exp"]:
            for node_name_1 in ["Exp"]:
                for axis in range(3):
                    node = helper.make_node(node_name_0, ["X"], ["Y"])
                    node2 = helper.make_node(node_name_1, ["Y"], ["Y1"])
                    argmax = helper.make_node(
                        "ArgMax", ["Y1"], ["Z"], axis=axis)
                    output_shape = self._get_argmax_output_shape(
                        input_shape, axis, True)
                    graph = helper.make_graph(
                        [node, node2, argmax],
                        "test",
                        [helper.make_tensor_value_info(
                            "X", TensorProto.FLOAT, input_shape)],
                        [helper.make_tensor_value_info("Z", TensorProto.INT64, output_shape)])
                    optimized_model = self._optimized(
                        graph, ["eliminate_nop_monotone_argmax"], True)
                    _assertion(graph, optimized_model, True, axis)
        # no axis X axis test
        for node_name_0 in ["Exp"]:
            for node_name_1 in ["Softmax", "LogSoftmax"]:
                for axis_0 in range(3):
                    for axis_1 in range(3):
                        node = helper.make_node(node_name_0, ["X"], ["Y"])
                        node2 = helper.make_node(
                            node_name_1, ["Y"], ["Y1"], axis=axis_0)
                        argmax = helper.make_node(
                            "ArgMax", ["Y1"], ["Z"], axis=axis_1)
                        output_shape = self._get_argmax_output_shape(
                            input_shape, axis_1, True)
                        graph = helper.make_graph(
                            [node, node2, argmax],
                            "test",
                            [helper.make_tensor_value_info(
                                "X", TensorProto.FLOAT, (5, 7, 11))],
                            [helper.make_tensor_value_info("Z", TensorProto.INT64, output_shape)])
                        optimized_model = self._optimized(
                            graph, ["eliminate_nop_monotone_argmax"], True)
                        _assertion(graph, optimized_model,
                                   axis_0 == axis_1, axis_1)
        # axis X axis test
        for node_name_0 in ["Softmax", "LogSoftmax"]:
            for node_name_1 in ["Softmax", "LogSoftmax"]:
                for axis_0 in range(3):
                    for axis_1 in range(3):
                        for axis_2 in range(3):
                            node = helper.make_node(
                                node_name_0, ["X"], ["Y"], axis=axis_0)
                            node2 = helper.make_node(
                                node_name_1, ["Y"], ["Y1"], axis=axis_1)
                            argmax = helper.make_node(
                                "ArgMax", ["Y1"], ["Z"], axis=axis_2)
                            output_shape = self._get_argmax_output_shape(
                                input_shape, axis_2, True)
                            graph = helper.make_graph(
                                [node, node2, argmax],
                                "test",
                                [helper.make_tensor_value_info(
                                    "X", TensorProto.FLOAT, input_shape)],
                                [helper.make_tensor_value_info("Z", TensorProto.INT64, output_shape)])
                            optimized_model = self._optimized(
                                graph, ["eliminate_nop_monotone_argmax"], True)
                            if axis_0 == axis_1:  # we can reduce both of the monotonic ops
                                _assertion(graph, optimized_model,
                                           axis_1 == axis_2, axis_2)
                            elif axis_1 == axis_2:  # we can reduce one of the monotonic ops
                                assert len(optimized_model.graph.output) == 1
                                assert len(optimized_model.graph.node) == 2
                                assert optimized_model.graph.output[0].type.tensor_type.elem_type == TensorProto.INT64
                                assert optimized_model.graph.node[-1].op_type == "ArgMax"
                                assert optimized_model.graph.node[-1].attribute[0].name == "axis"
                                assert optimized_model.graph.node[-1].attribute[0].i == axis_2
                            else:  # we can't reduce anything
                                assert optimized_model.graph == graph

    def test_eliminate_nop_dropout(self):  # type: () -> None
        node = helper.make_node("Dropout", ["X"], ["Y"])
        node1 = helper.make_node("Log", ["Y"], ["Z"])
        graph = helper.make_graph(
            [node, node1],
            "test",
            [helper.make_tensor_value_info(
                "X", TensorProto.FLOAT, (5, 7))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (5, 7))])
        optimized_model = self._optimized(
            graph, ["eliminate_nop_dropout"], False)

        # we don't want to eliminate the dropoutin opset 12,
        # even when it';s an optional parameter (defaults to 0)
        assert optimized_model.graph == graph

    # type: () -> None
    def test_eliminate_nop_dropout_opset11_graph_output(self):
        node = helper.make_node("Log", ["X"], ["Y"])
        node1 = helper.make_node("Dropout", ["Y"], ["Z"], ratio=0.0)
        graph = helper.make_graph(
            [node, node1],
            "test",
            [helper.make_tensor_value_info(
                "X", TensorProto.FLOAT, (5, 7))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (5, 7))])
        optimized_model = self._optimized(
            graph, ["eliminate_nop_dropout"], False, opset_imports=[helper.make_opsetid("", 11)])

        assert len(optimized_model.graph.output) == 1
        assert len(optimized_model.graph.node) == 1
        assert optimized_model.graph.node[0].op_type == "Log"
        assert optimized_model.graph.output[0].name == 'Z'

    def test_eliminate_nop_dropout_opset11(self):  # type: () -> None
        for ratio in [0.0, 0.5]:
            node = helper.make_node("Dropout", ["X"], ["Y"], ratio=ratio)
            node1 = helper.make_node("Log", ["Y"], ["Z"])
            graph = helper.make_graph(
                [node, node1],
                "test",
                [helper.make_tensor_value_info(
                    "X", TensorProto.FLOAT, (5, 7))],
                [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (5, 7))])
            optimized_model = self._optimized(
                graph, ["eliminate_nop_dropout"], False, opset_imports=[helper.make_opsetid("", 11)])

            if ratio > 0.0:
                assert optimized_model.graph == graph
            else:
                assert len(optimized_model.graph.output) == 1
                assert len(optimized_model.graph.node) == 1
                assert optimized_model.graph.node[0].op_type == "Log"

    def test_fuse_reduction_unsqueeze(self):  # type: () -> None
        # type: (Tuple[int, ...], List[int], List[int], bool) -> Tuple[int, ...]
        def _calculate_post_transform_shape(input_shape, reduction_axes, unsqueeze_axes, keepdim):
            post_reduce_shape = None
            if keepdim:
                post_reduce_shape = tuple(
                    [(x if i not in reduction_axes else 1) for i, x in enumerate(input_shape)])
            else:
                post_reduce_shape = tuple(
                    [x for i, x in enumerate(input_shape) if i not in reduction_axes])
            post_unsqueeze_shape = list(post_reduce_shape)
            for ax in unsqueeze_axes:
                post_unsqueeze_shape.insert(ax, 1)
            return tuple(post_unsqueeze_shape)

        for reduction in ["ReduceL1", "ReduceL2", "ReduceLogSum",
                          "ReduceLogSumExp", "ReduceMax", "ReduceMean",
                          "ReduceMin", "ReduceProd", "ReduceSum", "ReduceSumSquare"]:
            for axes1 in [[1], [1, 2], [2]]:
                for axes2 in [[0], [0, 1], [1]]:
                    for keepdim in [False, True]:
                        input_shape = (5, 7, 9)
                        output_shape = _calculate_post_transform_shape(
                            input_shape, axes1, axes2, keepdim)  # type: Tuple[int, ...]
                        axes2_arr = np.array(axes2, dtype=np.int64)
                        graph_input = [helper.make_tensor_value_info(
                            "X", TensorProto.FLOAT, input_shape),
                            helper.make_tensor_value_info("Y_axes", TensorProto.INT64, axes2_arr.shape)]
                        graph_initializer = [
                            helper.make_tensor("Y_axes", TensorProto.INT64,
                                               axes2_arr.shape, axes2_arr.tobytes(), raw=True)
                        ]
                        if reduction in ("ReduceSum"):
                            axes1_arr = np.array(axes1, dtype=np.int64)
                            node = helper.make_node(
                                reduction, ["X", "X_axes"], ["Y"], keepdims=keepdim)
                            graph_input.append(
                                helper.make_tensor_value_info("X_axes", TensorProto.INT64, axes1_arr.shape))
                            graph_initializer.append(helper.make_tensor("X_axes", TensorProto.INT64,
                                                                        axes1_arr.shape, axes1_arr.tobytes(), raw=True))
                        else:
                            node = helper.make_node(
                                reduction, ["X"], ["Y"], axes=axes1, keepdims=keepdim)

                        node1 = helper.make_node(
                            "Unsqueeze", ["Y", "Y_axes"], ["Z"])
                        graph = helper.make_graph(
                            [node, node1],
                            "test",
                            graph_input,
                            [helper.make_tensor_value_info(
                                "Z", TensorProto.FLOAT, output_shape)],
                            initializer=graph_initializer
                        )
                        optimized_model = self._optimized(
                            graph, ["fuse_consecutive_reduce_unsqueeze"], False)

                        if keepdim or axes1 != axes2:
                            assert optimized_model.graph == graph
                        else:
                            assert len(optimized_model.graph.output) == 1
                            assert len(optimized_model.graph.node) == 1
                            assert optimized_model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
                            assert optimized_model.graph.node[-1].op_type == reduction

                            if reduction in ("ReduceSum"):
                                for init in optimized_model.graph.initializer:
                                    if init.name == optimized_model.graph.node[-1].input[1]:
                                        assert list(to_array(init)) == axes1
                            else:
                                assert optimized_model.graph.node[-1].attribute[0].name == "axes"
                                assert optimized_model.graph.node[-1].attribute[0].ints == axes1
                            optimized_output_shape = tuple(
                                x.dim_value for x in optimized_model.graph.output[0].type.tensor_type.shape.dim)
                            assert optimized_output_shape == output_shape

    def test_split_predict_and_lift_lexical_references_for_caffe2_backend(self):
        model_str = b'\x08\x06\x12\x07pytorch\x1a\x031.9:\xe5\x02\n\'\x12\x011"\x08Constant*\x18\n\x05value*\x0c\x10\x07J\x08\x05\x00\x00\x00\x00\x00\x00\x00\xa0\x01\x04\n \x12\x012"\x08Constant*\x11\n\x05value*\x05\x10\tJ\x01\x01\xa0\x01\x04\n\xd1\x01\n\x011\n\x012\n\x03x.1\x12\x013"\x04Loop*\xba\x01\n\x04body2\xae\x01\n\x1a\n\x04x.11\n\x03i.1\x12\x017\x1a\x05Add_0"\x03Add\n\x1c\n\x012\x12\x018\x1a\nIdentity_1"\x08Identity\x12\x11torch-jit-export1Z\r\n\x03i.1\x12\x06\n\x04\x08\x07\x12\x00Z\x0e\n\x04cond\x12\x06\n\x04\x08\t\x12\x00Z\x1a\n\x04x.11\x12\x12\n\x10\x08\x07\x12\x0c\n\x02\x08\x01\n\x02\x08\x02\n\x02\x08\x03b\x0b\n\x018\x12\x06\n\x04\x08\t\x12\x00b\x17\n\x017\x12\x12\n\x10\x08\x07\x12\x0c\n\x02\x08\x01\n\x02\x08\x02\n\x02\x08\x03\xa0\x01\x05\x12\x10torch-jit-exportZ\x19\n\x03x.1\x12\x12\n\x10\x08\x07\x12\x0c\n\x02\x08\x01\n\x02\x08\x02\n\x02\x08\x03b\x17\n\x013\x12\x12\n\x10\x08\x07\x12\x0c\n\x02\x08\x01\n\x02\x08\x02\n\x02\x08\x03B\x02\x10\t'
        model = onnx.load_from_string(model_str)
        passes = ['fuse_consecutive_transposes',
                  'eliminate_nop_transpose',
                  'fuse_transpose_into_gemm',
                  'lift_lexical_references',
                  'split_predict']
        self._optimized(model, passes, fixed_point=True, compare_result=False, check=False)

    @unittest.skipUnless(has_tv, "This test needs torchvision")
    def test_torchvision_fasterrcnn_fpn(self):    # type: () -> None
        model = tv.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        with io.BytesIO() as f:
            torch.onnx.export(model, x, f, opset_version=11)
            model = onnx.load_model_from_string(f.getvalue())
            self._optimized(
                model, onnxoptimizer.get_fuse_and_elimination_passes(), fixed_point=True)

    # maskrcnn is only supported in opset 11 and higher
    @unittest.skipUnless(has_tv, "This test needs torchvision")
    def test_torchvision_maskrcnn_fpn_opset11(self):    # type: () -> None
        model = tv.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
        x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        with io.BytesIO() as f:
            torch.onnx.export(model, x, f, opset_version=11)
            model = onnx.load_model_from_string(f.getvalue())
            self._optimized(
                model, onnxoptimizer.get_fuse_and_elimination_passes(), fixed_point=True)

    # keypointrcnn is only supported in opset 11 and higher
    @unittest.skipUnless(has_tv, "This test needs torchvision")
    def test_torchvision_keypointrcnn_fpn(self):    # type: () -> None
        model = tv.models.detection.keypointrcnn_resnet50_fpn(pretrained=False)
        x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        with io.BytesIO() as f:
            torch.onnx.export(model, x, f, opset_version=11)
            model = onnx.load_model_from_string(f.getvalue())
            self._optimized(
                model, onnxoptimizer.get_fuse_and_elimination_passes(), fixed_point=True)

    @unittest.skipUnless(has_tv, "This test needs torchvision")
    def test_torchvision_shufflenet_v2(self):    # type: () -> None
        model = tv.models.shufflenet_v2_x1_0(pretrained=False)
        x = torch.rand(1, 3, 224, 224)
        with io.BytesIO() as f:
            torch.onnx.export(model, x, f)
            model = onnx.load_model_from_string(f.getvalue())
            self._optimized(
                model, onnxoptimizer.get_fuse_and_elimination_passes(), fixed_point=True)

    @unittest.skipUnless(has_tv, "This test needs torchvision")
    def test_torchvision_mnasnet(self):    # type: () -> None
        model = tv.models.mnasnet1_0(pretrained=False)
        x = torch.rand(1, 3, 224, 224)
        with io.BytesIO() as f:
            torch.onnx.export(model, x, f)
            model = onnx.load_model_from_string(f.getvalue())
            self._optimized(
                model, onnxoptimizer.get_fuse_and_elimination_passes(), fixed_point=True)

    @unittest.skipUnless(has_tv, "This test needs torchvision")
    def test_torchvision_deeplabv3(self):    # type: () -> None
        model = tv.models.segmentation.deeplabv3_resnet50(pretrained=False)
        x = torch.rand(1, 3, 224, 224)
        with io.BytesIO() as f:
            torch.onnx.export(model, x, f)
            model = onnx.load_model_from_string(f.getvalue())
            self._optimized(
                model, onnxoptimizer.get_fuse_and_elimination_passes(), fixed_point=True)


if __name__ == '__main__':
    unittest.main()
