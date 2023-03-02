import argparse
import ast
import uvicorn


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'tools',
        choices=['compress', 'convert', 'simple_serving', 'paddle2coreml'])
    ## argumentments for auto compression
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help="path of compression strategy config.")
    parser.add_argument(
        '--method',
        type=str,
        default=None,
        help="choose PTQ or QAT as quantization method")
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./output',
        help="directory to save model.")
    parser.add_argument(
        '--devices',
        type=str,
        default='gpu',
        help="which device used to compress.")
    ## arguments for other x2paddle
    parser.add_argument(
        '--framework',
        type=str,
        default=None,
        help="define which deeplearning framework(tensorflow/caffe/onnx)")
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help="define model file path for tensorflow or onnx")
    parser.add_argument(
        "--prototxt",
        "-p",
        type=str,
        default=None,
        help="prototxt file of caffe model")
    parser.add_argument(
        "--weight",
        "-w",
        type=str,
        default=None,
        help="weight file of caffe model")
    parser.add_argument(
        "--caffe_proto",
        "-c",
        type=str,
        default=None,
        help="optional: the .py file compiled by caffe proto file of caffe model"
    )
    parser.add_argument(
       "--input_shape_dict",
       "-isd",
       type=str,
       default=None,
       help="define input shapes, e.g --input_shape_dict=\"{'image':[1, 3, 608, 608]}\" or" \
       "--input_shape_dict=\"{'image':[1, 3, 608, 608], 'im_shape': [1, 2], 'scale_factor': [1, 2]}\"")
    parser.add_argument(
        "--enable_code_optim",
        "-co",
        type=ast.literal_eval,
        default=False,
        help="Turn on code optimization")
    ## arguments for simple serving
    parser.add_argument(
        "--app",
        type=str,
        default="server:app",
        help="Simple serving app string")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Simple serving host IP address")
    parser.add_argument(
        "--port", type=int, default=8000, help="Simple serving host port")
    ## arguments for paddle2coreml
    parser.add_argument(
        "--p2c_paddle_model_dir",
        type=str,
        default=None,
        help="define paddle model path")
    parser.add_argument(
        "--p2c_coreml_model_dir",
        type=str,
        default=None,
        help="define generated coreml model path")
    parser.add_argument(
        "--p2c_coreml_model_name",
        type=str,
        default="coreml_model",
        help="define generated coreml model name")
    parser.add_argument(
        "--p2c_input_names", type=str, default=None, help="define input names")
    parser.add_argument(
        "--p2c_input_dtypes",
        type=str,
        default="float32",
        help="define input dtypes")
    parser.add_argument(
        "--p2c_input_shapes",
        type=str,
        default=None,
        help="define input shapes")
    parser.add_argument(
        "--p2c_output_names",
        type=str,
        default=None,
        help="define output names")
    ## arguments for other tools
    return parser


def main():
    args = argsparser().parse_args()
    if args.tools == "compress":
        from .auto_compression.fd_auto_compress.fd_auto_compress import auto_compress
        print("Welcome to use FastDeploy Auto Compression Toolkit!")
        auto_compress(args)
    if args.tools == "convert":
        try:
            import platform
            import logging
            v0, v1, v2 = platform.python_version().split('.')
            if not (int(v0) >= 3 and int(v1) >= 5):
                logging.info("[ERROR] python>=3.5 is required")
                return
            import paddle
            v0, v1, v2 = paddle.__version__.split('.')
            logging.info("paddle.__version__ = {}".format(paddle.__version__))
            if v0 == '0' and v1 == '0' and v2 == '0':
                logging.info(
                    "[WARNING] You are use develop version of paddlepaddle")
            elif int(v0) != 2 or int(v1) < 0:
                logging.info("[ERROR] paddlepaddle>=2.0.0 is required")
                return
            from x2paddle.convert import tf2paddle, caffe2paddle, onnx2paddle
            if args.framework == "tensorflow":
                assert args.model is not None, "--model should be defined while convert tensorflow model"
                tf2paddle(args.model, args.save_dir)
            elif args.framework == "caffe":
                assert args.prototxt is not None and args.weight is not None, "--prototxt and --weight should be defined while convert caffe model"
                caffe2paddle(args.prototxt, args.weight, args.save_dir,
                             args.caffe_proto)
            elif args.framework == "onnx":
                assert args.model is not None, "--model should be defined while convert onnx model"
                onnx2paddle(
                    args.model,
                    args.save_dir,
                    input_shape_dict=args.input_shape_dict)
            else:
                raise Exception(
                    "--framework only support tensorflow/caffe/onnx now")
        except ImportError:
            print(
                "Model convert failed! Please check if you have installed it!")
    if args.tools == "simple_serving":
        custom_logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": "uvicorn.logging.DefaultFormatter",
                    "fmt": "%(asctime)s %(levelprefix)s %(message)s",
                    'datefmt': '%Y-%m-%d %H:%M:%S',
                    "use_colors": None,
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                'null': {
                    "formatter": "default",
                    "class": 'logging.NullHandler'
                }
            },
            "loggers": {
                "": {
                    "handlers": ["null"],
                    "level": "DEBUG"
                },
                "uvicorn.error": {
                    "handlers": ["default"],
                    "level": "DEBUG"
                }
            },
        }
        uvicorn.run(args.app,
                    host=args.host,
                    port=args.port,
                    app_dir='.',
                    log_config=custom_logging_config)
    if args.tools == "paddle2coreml":
        if any([
                args.p2c_paddle_model_dir is None,
                args.p2c_coreml_model_dir is None,
                args.p2c_input_names is None, args.p2c_input_shapes is None,
                args.p2c_output_names is None
        ]):
            raise Exception(
                "paddle2coreml need to define --p2c_paddle_model_dir, --p2c_coreml_model_dir, --p2c_input_names, --p2c_input_shapes, --p2c_output_names"
            )
        import coremltools as ct
        import os
        import numpy as np

        def type_to_np_dtype(dtype):
            if dtype == 'float32':
                return np.float32
            elif dtype == 'float64':
                return np.float64
            elif dtype == 'int32':
                return np.int32
            elif dtype == 'int64':
                return np.int64
            elif dtype == 'uint8':
                return np.uint8
            elif dtype == 'uint16':
                return np.uint16
            elif dtype == 'uint32':
                return np.uint32
            elif dtype == 'uint64':
                return np.uint64
            elif dtype == 'int8':
                return np.int8
            elif dtype == 'int16':
                return np.int16
            else:
                raise Exception("Unsupported dtype: {}".format(dtype))

        input_names = args.p2c_input_names.split(' ')
        input_shapes = [[int(i) for i in shape.split(',')]
                        for shape in args.p2c_input_shapes.split(' ')]
        input_dtypes = map(type_to_np_dtype, args.p2c_input_dtypes.split(' '))
        output_names = args.p2c_output_names.split(' ')
        sample_input = [
            ct.TensorType(
                name=k,
                shape=s,
                dtype=d, )
            for k, s, d in zip(input_names, input_shapes, input_dtypes)
        ]

        coreml_model = ct.convert(
            args.p2c_paddle_model_dir,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS13,
            inputs=sample_input,
            outputs=[ct.TensorType(name=name) for name in output_names], )
        coreml_model.save(
            os.path.join(args.p2c_coreml_model_dir,
                         args.p2c_coreml_model_name))


if __name__ == '__main__':
    main()
