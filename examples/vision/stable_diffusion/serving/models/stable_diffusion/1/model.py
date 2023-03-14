# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import json
import io
import base64

import fastdeploy as fd
import numpy as np
import paddle
from fastdeploy import ModelFormat

from paddlenlp.trainer.argparser import strtobool
from paddlenlp.transformers import CLIPTokenizer
from ppdiffusers import (
    FastDeployRuntimeModel,
    FastDeployStableDiffusionPipeline,
    PNDMScheduler,
    PreconfigEulerAncestralDiscreteScheduler, )

import triton_python_backend_utils as pb_utils

dir_name = os.path.dirname(os.path.realpath(__file__)) + "/"


class MyArgs:
    def __init__(self):
        self.scheduler = 'pndm'
        self.device_id = 0
        self.use_fp16 = True
        self.device = 'gpu'
        self.backend = 'paddle_tensorrt'
        self.inference_steps = 50
        self.model_dir = dir_name + 'stable-diffusion-v1-5/'
        self.model_format = 'paddle'
        self.unet_model_prefix = 'unet'
        self.vae_decoder_model_prefix = 'vae_decoder'
        self.text_encoder_model_prefix = 'text_encoder'


def image2byte(image):
    img_bytes = io.BytesIO()
    image = image.convert("RGB")
    image.save(img_bytes, format="JPEG")
    image_bytes = img_bytes.getvalue()
    return base64.b64encode(image_bytes)


def create_paddle_inference_runtime(model_dir,
                                    model_prefix,
                                    use_trt=False,
                                    dynamic_shape=None,
                                    use_fp16=False,
                                    device_id=0,
                                    disable_paddle_trt_ops=[],
                                    disable_paddle_pass=[],
                                    paddle_stream=None):

    option = fd.RuntimeOption()
    option.use_paddle_backend()
    if device_id == -1:
        option.use_cpu()
    else:
        option.use_gpu(device_id)
    if paddle_stream is not None:
        option.set_external_raw_stream(paddle_stream)
    for pass_name in disable_paddle_pass:
        option.paddle_infer_option.delete_pass(pass_name)
    if use_trt:
        option.paddle_infer_option.disable_trt_ops(disable_paddle_trt_ops)
        option.paddle_infer_option.enable_trt = True
        if use_fp16:
            option.trt_option.enable_fp16 = True
        cache_file = os.path.join(model_dir, model_prefix, "_opt_cache/")
        option.set_trt_cache_file(cache_file)
        # Need to enable collect shape for ernie
        if dynamic_shape is not None:
            option.enable_paddle_trt_collect_shape()
            for key, shape_dict in dynamic_shape.items():
                option.set_trt_input_shape(
                    key,
                    min_shape=shape_dict["min_shape"],
                    opt_shape=shape_dict.get("opt_shape", None),
                    max_shape=shape_dict.get("max_shape", None), )
    model_file = os.path.join(model_dir, model_prefix, "inference.pdmodel")
    params_file = os.path.join(model_dir, model_prefix, "inference.pdiparams")
    option.set_model_path(model_file, params_file)
    return fd.Runtime(option)


def get_scheduler(args):
    if args.scheduler == "pndm":
        scheduler = PNDMScheduler(
            beta_end=0.012,
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            num_train_timesteps=1000,
            skip_prk_steps=True,
            steps_offset=1, )
    elif args.scheduler == "euler_ancestral":
        scheduler = PreconfigEulerAncestralDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            preconfig=True)
    else:
        raise ValueError(
            f"Scheduler '{args.scheduler}' is not supportted right now.")
    return scheduler


def create_pipe(args):
    # 0. Init device id
    device_id = args.device_id
    if args.device == "cpu":
        device_id = -1
        paddle.set_device("cpu")
        paddle_stream = None
    else:
        paddle.set_device(f"gpu:{device_id}")
        paddle_stream = paddle.device.cuda.current_stream(
            device_id).cuda_stream
    # 1. Init scheduler
    scheduler = get_scheduler(args)

    # 2. Init tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        os.path.join(args.model_dir, "tokenizer"))

    # 3. Set dynamic shape for trt backend
    text_encoder_shape = {
        "input_ids": {
            "min_shape": [1, 77],
            "max_shape": [2, 77],
            "opt_shape": [1, 77],
        }
    }
    vae_dynamic_shape = {
        "latent_sample": {
            "min_shape": [1, 4, 64, 64],
            "max_shape": [1, 4, 64, 64],
            "opt_shape": [1, 4, 64, 64],
        }
    }

    unet_dynamic_shape = {
        "sample": {
            "min_shape": [1, 4, 64, 64],
            "max_shape": [2, 4, 64, 64],
            "opt_shape": [2, 4, 64, 64],
        },
        "timestep": {
            "min_shape": [1],
            "max_shape": [1],
            "opt_shape": [1],
        },
        "encoder_hidden_states": {
            "min_shape": [1, 77, 768],
            "max_shape": [2, 77, 768],
            "opt_shape": [2, 77, 768],
        },
    }

    use_trt = True if args.backend == "paddle_tensorrt" else False
    start = time.time()
    unet_runtime = create_paddle_inference_runtime(
        args.model_dir,
        args.unet_model_prefix,
        use_trt,
        unet_dynamic_shape,
        use_fp16=args.use_fp16,
        device_id=args.device_id,
        paddle_stream=paddle_stream, )
    print(f"Spend {time.time() - start : .2f} s to load unet model.")

    text_encoder_runtime = create_paddle_inference_runtime(
        args.model_dir,
        args.text_encoder_model_prefix,
        use_trt,
        text_encoder_shape,
        use_fp16=args.use_fp16,
        device_id=device_id,
        disable_paddle_trt_ops=["arg_max", "range", "lookup_table_v2"],
        paddle_stream=paddle_stream, )
    vae_decoder_runtime = create_paddle_inference_runtime(
        args.model_dir,
        args.vae_decoder_model_prefix,
        use_trt,
        vae_dynamic_shape,
        use_fp16=args.use_fp16,
        device_id=device_id,
        paddle_stream=paddle_stream, )

    pipe = FastDeployStableDiffusionPipeline(
        vae_encoder=None,
        vae_decoder=FastDeployRuntimeModel(model=vae_decoder_runtime),
        text_encoder=FastDeployRuntimeModel(model=text_encoder_runtime),
        tokenizer=tokenizer,
        unet=FastDeployRuntimeModel(model=unet_runtime),
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None, )
    # Warm up
    prompt = "a photo of an astronaut riding a horse on mars"
    pipe.scheduler.set_timesteps(10)
    pipe(prompt, num_inference_steps=10)
    pipe.scheduler.set_timesteps(args.inference_steps)
    return pipe


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        self.model_args = MyArgs()
        if args['model_instance_kind'] == "GPU":
            self.model_args.device_id = int(args['model_instance_device_id'][
                0])
        self.pipe = create_pipe(self.model_args)

        # You must parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args['model_config'])
        print("model_config:", self.model_config)

        self.input_names = []
        for input_config in self.model_config["input"]:
            self.input_names.append(input_config["name"])
        print("input:", self.input_names)

        self.output_names = []
        self.output_dtype = []
        for output_config in self.model_config["output"]:
            self.output_names.append(output_config["name"])
            dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
            self.output_dtype.append(dtype)
        print("output:", self.output_names)

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []
        prompt = []
        for request in requests:
            data = pb_utils.get_input_tensor_by_name(request,
                                                     self.input_names[0])
            data = data.as_numpy()
            data = [i[0].decode('utf-8') for i in data]
            prompt.append(data[0])
        # batch predict, for now, only support batch = 1
        images = self.pipe(
            prompt, num_inference_steps=self.model_args.inference_steps).images
        for image in images:
            image_byte = image2byte(image)
            out_result = np.array(image_byte, dtype='object')
            out_tensor = pb_utils.Tensor(self.output_names[0], out_result)
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                out_tensor,
            ])
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
