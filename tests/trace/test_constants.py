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

from fastdeploy.trace.constants import EVENT_TO_STAGE_MAP, LoggingEventName, StageName


class TestLoggingEventName:
    def test_enum_values(self):
        assert LoggingEventName.PREPROCESSING_START.value == "PREPROCESSING_START"
        assert LoggingEventName.PREPROCESSING_END.value == "PREPROCESSING_END"
        assert LoggingEventName.REQUEST_SCHEDULE_START.value == "REQUEST_SCHEDULE_START"
        assert LoggingEventName.REQUEST_QUEUE_START.value == "REQUEST_QUEUE_START"
        assert LoggingEventName.REQUEST_QUEUE_END.value == "REQUEST_QUEUE_END"
        assert LoggingEventName.RESOURCE_ALLOCATE_START.value == "RESOURCE_ALLOCATE_START"
        assert LoggingEventName.RESOURCE_ALLOCATE_END.value == "RESOURCE_ALLOCATE_END"
        assert LoggingEventName.REQUEST_SCHEDULE_END.value == "REQUEST_SCHEDULE_END"
        assert LoggingEventName.INFERENCE_START.value == "INFERENCE_START"
        assert LoggingEventName.FIRST_TOKEN_GENERATED.value == "FIRST_TOKEN_GENERATED"
        assert LoggingEventName.DECODE_START.value == "DECODE_START"
        assert LoggingEventName.INFERENCE_END.value == "INFERENCE_END"
        assert LoggingEventName.POSTPROCESSING_START.value == "POSTPROCESSING_START"
        assert LoggingEventName.POSTPROCESSING_END.value == "POSTPROCESSING_END"


class TestStageName:
    def test_enum_values(self):
        assert StageName.PREPROCESSING.value == "PREPROCESSING"
        assert StageName.SCHEDULE.value == "SCHEDULE"
        assert StageName.PREFILL.value == "PREFILL"
        assert StageName.DECODE.value == "DECODE"
        assert StageName.POSTPROCESSING.value == "POSTPROCESSING"


class TestEventToStageMap:
    def test_mapping(self):
        assert EVENT_TO_STAGE_MAP[LoggingEventName.PREPROCESSING_START] == StageName.PREPROCESSING
        assert EVENT_TO_STAGE_MAP[LoggingEventName.PREPROCESSING_END] == StageName.PREPROCESSING
        assert EVENT_TO_STAGE_MAP[LoggingEventName.REQUEST_SCHEDULE_START] == StageName.SCHEDULE
        assert EVENT_TO_STAGE_MAP[LoggingEventName.REQUEST_QUEUE_START] == StageName.SCHEDULE
        assert EVENT_TO_STAGE_MAP[LoggingEventName.REQUEST_QUEUE_END] == StageName.SCHEDULE
        assert EVENT_TO_STAGE_MAP[LoggingEventName.RESOURCE_ALLOCATE_START] == StageName.SCHEDULE
        assert EVENT_TO_STAGE_MAP[LoggingEventName.RESOURCE_ALLOCATE_END] == StageName.SCHEDULE
        assert EVENT_TO_STAGE_MAP[LoggingEventName.REQUEST_SCHEDULE_END] == StageName.SCHEDULE
        assert EVENT_TO_STAGE_MAP[LoggingEventName.INFERENCE_START] == StageName.PREFILL
        assert EVENT_TO_STAGE_MAP[LoggingEventName.FIRST_TOKEN_GENERATED] == StageName.PREFILL
        assert EVENT_TO_STAGE_MAP[LoggingEventName.DECODE_START] == StageName.DECODE
        assert EVENT_TO_STAGE_MAP[LoggingEventName.INFERENCE_END] == StageName.DECODE
        assert EVENT_TO_STAGE_MAP[LoggingEventName.POSTPROCESSING_START] == StageName.POSTPROCESSING
        assert EVENT_TO_STAGE_MAP[LoggingEventName.POSTPROCESSING_END] == StageName.POSTPROCESSING
