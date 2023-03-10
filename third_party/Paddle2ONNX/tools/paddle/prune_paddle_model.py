import argparse


def new_prepend_feed_ops(inference_program,
                         feed_target_names,
                         feed_holder_name='feed'):
    import paddle.fluid.core as core
    if len(feed_target_names) == 0:
        return

    global_block = inference_program.global_block()
    feed_var = global_block.create_var(
        name=feed_holder_name,
        type=core.VarDesc.VarType.FEED_MINIBATCH,
        persistable=True)

    for i, name in enumerate(feed_target_names):
        if not global_block.has_var(name):
            print(
                "The input[{i}]: '{name}' doesn't exist in pruned inference program, which will be ignored in new saved model.".
                format(
                    i=i, name=name))
            continue
        out = global_block.var(name)
        global_block._prepend_op(
            type='feed',
            inputs={'X': [feed_var]},
            outputs={'Out': [out]},
            attrs={'col': i})


def append_fetch_ops(program, fetch_target_names, fetch_holder_name='fetch'):
    """
    In this palce, we will add the fetch op
    """
    import paddle.fluid.core as core
    global_block = program.global_block()
    fetch_var = global_block.create_var(
        name=fetch_holder_name,
        type=core.VarDesc.VarType.FETCH_LIST,
        persistable=True)
    print("the len of fetch_target_names:%d" % (len(fetch_target_names)))
    for i, name in enumerate(fetch_target_names):

        global_block.append_op(
            type='fetch',
            inputs={'X': [name]},
            outputs={'Out': [fetch_var]},
            attrs={'col': i})


def insert_fetch(program, fetchs, fetch_holder_name="fetch"):
    global_block = program.global_block()
    need_to_remove_op_index = list()
    for i, op in enumerate(global_block.ops):
        if op.type == 'fetch':
            need_to_remove_op_index.append(i)
    for index in need_to_remove_op_index[::-1]:
        global_block._remove_op(index)
    program.desc.flush()
    append_fetch_ops(program, fetchs, fetch_holder_name)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        required=True,
        help='Path of directory saved the input model.')
    parser.add_argument(
        '--model_filename', required=True, help='The input model file name.')
    parser.add_argument(
        '--params_filename', required=True, help='The parameters file name.')
    parser.add_argument(
        '--output_names',
        required=True,
        nargs='+',
        help='The outputs of pruned model.')
    parser.add_argument(
        '--save_dir',
        required=True,
        help='Path of directory to save the new exported model.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    if len(set(args.output_names)) < len(args.output_names):
        print(
            "[ERROR] There's dumplicate name in --output_names, which is not allowed."
        )
        sys.exit(-1)

    import paddle
    paddle.enable_static()
    paddle.fluid.io.prepend_feed_ops = new_prepend_feed_ops
    import paddle.fluid as fluid
    print("Start to load paddle model...")
    exe = fluid.Executor(fluid.CPUPlace())
    [prog, ipts, outs] = fluid.io.load_inference_model(
        args.model_dir,
        exe,
        model_filename=args.model_filename,
        params_filename=args.params_filename)
    new_outputs = list()
    insert_fetch(prog, args.output_names)
    for out_name in args.output_names:
        new_outputs.append(prog.global_block().var(out_name))
    fluid.io.save_inference_model(
        args.save_dir,
        ipts,
        new_outputs,
        exe,
        prog,
        model_filename=args.model_filename,
        params_filename=args.params_filename)
