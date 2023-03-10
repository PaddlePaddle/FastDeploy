#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

OP_MAPPING_WAITTING = 0
OP_MAPPING_NO_REGISTER = 1
OP_MAPPING_NOT_STANDARD = 2

OP_MAPPING_IDENTITY = 10     # op is mapped directly, normally using convert_identity_operation
OP_MAPPING_WITH_EXTRA = 11   # op is mapped into multiply caffe nodes
OP_MAPPING_WITH_FUSED = 12   # op is mapped into less caffe nodes, properly process other ops along with current op
OP_MAPPING_AMMEND = 13       # op is cut down and re-connected, properly because op is meaningless(cast, scale with 1.)
OP_MAPPING_SKIPPED = 14      # op is skipped, properly get processed in former mapping, eg. OP_MAPPING_WITH_FUSED
OP_MAPPING_PENDDING = 15     # op is pending, properly wait for process in later mapping. eg. OP_MAPPING_WITH_EXTRA
OP_MAPPING_ABORT = 16        # [attention] op along with its sub op is all abort, only used by early stop
