# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from enum import Enum


class LoggingEventName(Enum):
    """
    Represents various event points in the system.
    """

    PREPROCESSING_START = "PREPROCESSING_START"
    PREPROCESSING_END = "PREPROCESSING_END"
    REQUEST_SCHEDULE_START = "REQUEST_SCHEDULE_START"
    REQUEST_QUEUE_START = "REQUEST_QUEUE_START"
    REQUEST_QUEUE_END = "REQUEST_QUEUE_END"
    RESOURCE_ALLOCATE_START = "RESOURCE_ALLOCATE_START"
    RESOURCE_ALLOCATE_END = "RESOURCE_ALLOCATE_END"
    REQUEST_SCHEDULE_END = "REQUEST_SCHEDULE_END"
    INFERENCE_START = "INFERENCE_START"
    FIRST_TOKEN_GENERATED = "FIRST_TOKEN_GENERATED"
    DECODE_START = "DECODE_START"
    INFERENCE_END = "INFERENCE_END"
    POSTPROCESSING_START = "POSTPROCESSING_START"
    POSTPROCESSING_END = "POSTPROCESSING_END"


class StageName(Enum):
    """
    Represents the main stages in the request processing flow.
    """

    PREPROCESSING = "PREPROCESSING"
    SCHEDULE = "SCHEDULE"
    PREFILL = "PREFILL"
    DECODE = "DECODE"
    POSTPROCESSING = "POSTPROCESSING"


EVENT_TO_STAGE_MAP = {
    LoggingEventName.PREPROCESSING_START: StageName.PREPROCESSING,
    LoggingEventName.PREPROCESSING_END: StageName.PREPROCESSING,
    LoggingEventName.REQUEST_SCHEDULE_START: StageName.SCHEDULE,
    LoggingEventName.REQUEST_QUEUE_START: StageName.SCHEDULE,
    LoggingEventName.REQUEST_QUEUE_END: StageName.SCHEDULE,
    LoggingEventName.RESOURCE_ALLOCATE_START: StageName.SCHEDULE,
    LoggingEventName.RESOURCE_ALLOCATE_END: StageName.SCHEDULE,
    LoggingEventName.REQUEST_SCHEDULE_END: StageName.SCHEDULE,
    LoggingEventName.INFERENCE_START: StageName.PREFILL,
    LoggingEventName.FIRST_TOKEN_GENERATED: StageName.PREFILL,
    LoggingEventName.DECODE_START: StageName.DECODE,
    LoggingEventName.INFERENCE_END: StageName.DECODE,
    LoggingEventName.POSTPROCESSING_START: StageName.POSTPROCESSING,
    LoggingEventName.POSTPROCESSING_END: StageName.POSTPROCESSING,
}
