import argparse


def process_old_ops_desc(program):
    for i in range(len(program.blocks[0].ops)):
        if program.blocks[0].ops[i].type == "matmul":
            if not program.blocks[0].ops[i].has_attr("head_number"):
                program.blocks[0].ops[i]._set_attr("head_number", 1)


def infer_shape(program, input_shape_dict):
    import paddle
    paddle.enable_static()
    import paddle.fluid as fluid

    OP_WITHOUT_KERNEL_SET = {
        'feed', 'fetch', 'recurrent', 'go', 'rnn_memory_helper_grad',
        'conditional_block', 'while', 'send', 'recv', 'listen_and_serv',
        'fl_listen_and_serv', 'ncclInit', 'select', 'checkpoint_notify',
        'gen_bkcl_id', 'c_gen_bkcl_id', 'gen_nccl_id', 'c_gen_nccl_id',
        'c_comm_init', 'c_sync_calc_stream', 'c_sync_comm_stream',
        'queue_generator', 'dequeue', 'enqueue', 'heter_listen_and_serv',
        'c_wait_comm', 'c_wait_compute', 'c_gen_hccl_id', 'c_comm_init_hccl',
        'copy_cross_scope'
    }
    model_version = program.desc._version()
    paddle_version = paddle.__version__
    major_ver = model_version // 1000000
    minor_ver = (model_version - major_ver * 1000000) // 1000
    patch_ver = model_version - major_ver * 1000000 - minor_ver * 1000
    model_version = "{}.{}.{}".format(major_ver, minor_ver, patch_ver)
    if model_version != paddle_version:
        print(
            "[WARNING] The model is saved by paddlepaddle v{}, but now your paddlepaddle is version of {}, this difference may cause error, it is recommend you reinstall a same version of paddlepaddle for this model".
            format(model_version, paddle_version))
    for k, v in input_shape_dict.items():
        program.blocks[0].var(k).desc.set_shape(v)
    for i in range(len(program.blocks)):
        for j in range(len(program.blocks[0].ops)):
            if program.blocks[i].ops[j].type in OP_WITHOUT_KERNEL_SET:
                continue
            program.blocks[i].ops[j].desc.infer_shape(program.blocks[i].desc)


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
        '--save_dir',
        required=True,
        help='Path of directory to save the new exported model.')
    parser.add_argument(
        '--input_shape_dict', required=True, help="The new shape information.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    import paddle
    paddle.enable_static()
    import paddle.fluid as fluid
    input_shape_dict_str = args.input_shape_dict
    input_shape_dict = eval(input_shape_dict_str)
    print("Start to load paddle model...")
    exe = fluid.Executor(fluid.CPUPlace())
    [prog, ipts, outs] = fluid.io.load_inference_model(
        args.model_dir,
        exe,
        model_filename=args.model_filename,
        params_filename=args.params_filename)
    process_old_ops_desc(prog)
    infer_shape(prog, input_shape_dict)
    fluid.io.save_inference_model(
        args.save_dir,
        ipts,
        outs,
        exe,
        prog,
        model_filename=args.model_filename,
        params_filename=args.params_filename)
