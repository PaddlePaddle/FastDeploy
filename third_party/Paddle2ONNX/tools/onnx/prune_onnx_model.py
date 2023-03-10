import argparse
import sys


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        required=True,
        help='Path of directory saved the input model.')
    parser.add_argument(
        '--output_names',
        required=True,
        nargs='+',
        help='The outputs of pruned model.')
    parser.add_argument(
        '--save_file', required=True, help='Path to save the new onnx model.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    import onnx
    model = onnx.load(args.model)
    output_tensor_names = set()
    for node in model.graph.node:
        for out in node.output:
            output_tensor_names.add(out)

    for output_name in args.output_names:
        if output_name not in output_tensor_names:
            print(
                "[ERROR] Cannot find output tensor name '{}' in onnx model graph.".
                format(output_name))
            sys.exit(-1)
    if len(set(args.output_names)) < len(args.output_names):
        print(
            "[ERROR] There's dumplicate name in --output_names, which is not allowed."
        )
        sys.exit(-1)

    output_node_indices = set()
    output_to_node = dict()
    for i, node in enumerate(model.graph.node):
        for out in node.output:
            output_to_node[out] = i
            if out in args.output_names:
                output_node_indices.add(i)

    # from outputs find all the ancestors
    import copy
    reserved_node_indices = copy.deepcopy(output_node_indices)
    reserved_inputs = set()
    new_output_node_indices = copy.deepcopy(output_node_indices)
    while True and len(new_output_node_indices) > 0:
        output_node_indices = copy.deepcopy(new_output_node_indices)
        new_output_node_indices = set()
        for out_node_idx in output_node_indices:
            for ipt in model.graph.node[out_node_idx].input:
                if ipt in output_to_node:
                    reserved_node_indices.add(output_to_node[ipt])
                    new_output_node_indices.add(output_to_node[ipt])
                else:
                    reserved_inputs.add(ipt)

    num_inputs = len(model.graph.input)
    num_outputs = len(model.graph.output)
    num_nodes = len(model.graph.node)
    print(len(reserved_node_indices), "xxxx")
    for idx in range(num_nodes - 1, -1, -1):
        if idx not in reserved_node_indices:
            del model.graph.node[idx]
    for idx in range(num_inputs - 1, -1, -1):
        if model.graph.input[idx].name not in reserved_inputs:
            del model.graph.input[idx]
    for out in args.output_names:
        model.graph.output.extend([onnx.ValueInfoProto(name=out)])
    for i in range(num_outputs):
        del model.graph.output[0]

    from onnx_infer_shape import SymbolicShapeInference
    model = SymbolicShapeInference.infer_shapes(model, 2**31 - 1, True, False,
                                                1)
    onnx.checker.check_model(model)
    onnx.save(model, args.save_file)
    print("[Finished] The new model saved in {}.".format(args.save_file))
    print("[DEBUG INFO] The inputs of new model: {}".format(
        [x.name for x in model.graph.input]))
    print("[DEBUG INFO] The outputs of new model: {}".format(
        [x.name for x in model.graph.output]))
