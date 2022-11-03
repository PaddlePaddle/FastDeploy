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

import json
import numpy as np
import time
import os
import sys
import codecs
import fastdeploy as fd

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


def load_dict():
    dir_name = os.path.dirname(os.path.realpath(__file__)) + "/"
    file_name = dir_name + "ppocr_keys_v1.txt"
    with open(file_name, encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]
        lines.insert(0, "#")
        lines.append(" ")
        return lines


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
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        print(sys.getdefaultencoding())
        # You must parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args['model_config'])
        print("model_config:", self.model_config)

        self.input_names = []
        for input_config in self.model_config["input"]:
            self.input_names.append(input_config["name"])
        print("postprocess input names:", self.input_names)

        self.output_names = []
        self.output_dtype = []
        for output_config in self.model_config["output"]:
            self.output_names.append(output_config["name"])
            dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
            self.output_dtype.append(dtype)
        print("postprocess output names:", self.output_names)
        self.label_list = load_dict()

    def postprocess(self, infer_outputs):
        """
        Parameters
        ----------
        infer_outputs : numpy.array
          Contains the batch of inference results
        im_infos : numpy.array(b'{}')
         Returns
        -------
        numpy.array
           postprocess result
        """
        results = []
        for i_batch in range(len(infer_outputs)):
            new_infer_output = infer_outputs[i_batch:i_batch + 1]
            #print(im_infos[i_batch])
            #new_im_info = im_infos[i_batch][0].decode('utf-8').replace("'", '"')
            #new_im_info = json.loads(new_im_info)

            result = fd.vision.ocr.Recognizer.postprocess([new_infer_output],
                                                          self.label_list)
            #print('result',type(result),result)
            #r_str = fd.vision.utils.fd_result_to_json(result)
            results.append(result)
        return np.array(results, dtype=np.object)

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
        # print("num:", len(requests), flush=True)
        for request in requests:
            infer_outputs = pb_utils.get_input_tensor_by_name(
                request, self.input_names[0])
            infer_outputs = infer_outputs.as_numpy()
            results = self.postprocess(infer_outputs)
            out_tensor = pb_utils.Tensor(self.output_names[0], results)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor, ])
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
