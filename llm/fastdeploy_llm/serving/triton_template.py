# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

"""
Templates for Triton Inference Server Model Repository
"""

triton_config_template = """
backend: "python"
max_batch_size: 0
model_transaction_policy {{
  decoupled: True
}}
input [
  {{
    name: "IN"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }}
]
output [
  {{
    name: "OUT"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }}
]
instance_group [{{ kind: KIND_CPU }}]

parameters {{
  key: "MAX_BATCH_SIZE"
  value: {{
    string_value: "{}"
  }}
}}

parameters {{
  key: "STOP_THRESHOLD"
  value: {{
    string_value: "{}"
  }}
}}

parameters {{
  key: "MAX_SEQ_LEN"
  value: {{
    string_value: "{}"
  }}
}}

parameters {{
  key: "MAX_DEC_LEN"
  value: {{
    string_value: "{}"
  }}
}}

# Decode Strategy only work for dynamic graph model
parameters {{
  key: "DECODE_STRATEGY"
  value: {{
    string_value: "{}"
  }}
}}

# MP NUM only configurable for dynamic graph model
parameters {{
  key: "MP_NUM"
  value: {{
    string_value: "{}"
  }}
}}

# quant_bits only work for dynamic graph model
parameters {{
  key: "QUANT_BITS"
  value: {{
    string_value: "{}"
  }}
}}

parameters {{
  key: "EOS_TOKEN_ID"
  value: {{
    string_value: "{}"
  }}
}}

parameters {{
  key: "MAX_PREFIX_LEN"
  value: {{
    string_value: "0"
  }}
}}

parameters {{
  key: "DISABLE_DYNAMIC_BATCHING"
  value: {{
    string_value: "0"
  }}
}}

parameters {{
  key: "DISABLE_DYNAMIC_BATCHING"
  value: {{
    string_value: "0"
  }}
}}

parameters {{
  key: "MAX_QUEUE_NUM"
  value: {{
    string_value: "0"
  }}
}}

parameters {{
  key: "IS_PTUNING"
  value: {{
    string_value: "0"
  }}
}}

parameters {{
  key: "MODEL_PROMPT_DIR_PATH"
  value: {{
    string_value: "0"
  }}
}}


"""

model_template = """
from fastdeploy_llm.serving import TritonPythonModel

"""
