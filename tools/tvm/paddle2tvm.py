import paddle
import tvm
from tvm import relay
from tvm.contrib import graph_executor
import os
import argparse


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", default="./picodet_l_320_coco_lcnet/model")
    parser.add_argument(
        "--shape_dict",
        default={"image": [1, 3, 320, 320],
                 "scale_factor": [1, 2]})
    parser.add_argument("--tvm_save_name", default="tvm_model")
    parser.add_argument("--tvm_save_path", default="./tvm_save")
    args = parser.parse_args()
    return args


def read_model(model_path):
    return paddle.jit.load(model_path)


def paddle_to_tvm(paddle_model,
                  shape_dict,
                  tvm_save_name="tvm_model",
                  tvm_save_path="./tvm_save"):
    if isinstance(shape_dict, str):
        shape_dict = eval(shape_dict)
    mod, params = relay.frontend.from_paddle(paddle_model, shape_dict)
    # 这里首先在PC的CPU上进行测试 所以使用LLVM进行导出
    target = tvm.target.Target("llvm", host="llvm")
    dev = tvm.cpu(0)
    # 这里利用TVM构建出优化后模型的信息
    with tvm.transform.PassContext(opt_level=2):
        base_lib = relay.build_module.build(mod, target, params=params)
        if not os.path.exists(tvm_save_path):
            os.mkdir(tvm_save_path)
        lib_save_path = os.path.join(tvm_save_path, tvm_save_name + ".so")
        base_lib.export_library(lib_save_path)
        param_save_path = os.path.join(tvm_save_path,
                                       tvm_save_name + ".params")
        with open(param_save_path, 'wb') as fo:
            fo.write(relay.save_param_dict(base_lib.get_params()))
        module = graph_executor.GraphModule(base_lib['default'](dev))
        module.load_params(relay.save_param_dict(base_lib.get_params()))
        print("export success")


if __name__ == "__main__":
    config = get_config()
    paddle_model = read_model(config.model_path)
    shape_dict = config.shape_dict
    paddle_to_tvm(paddle_model, shape_dict, config.tvm_save_name,
                  config.tvm_save_path)
