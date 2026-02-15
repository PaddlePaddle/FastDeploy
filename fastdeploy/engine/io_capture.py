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

"""
Engine I/O Capture Module

This module provides utilities to capture and dump EngineService input/output
for testing and verification between old and new architectures.
"""

import json
import os
import pickle
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class IOTypes:
    """I/O types for categorization."""
    REQUEST = "request"
    REQUEST_OUTPUT = "request_output"
    SCHEDULE_TASK = "schedule_task"
    WORKER_TASK = "worker_task"


class EngineIOCapture:
    """Capture engine service I/O for testing and verification.

    This class provides methods to capture requests, outputs, and tasks
    sent between EngineService, Scheduler, and Worker processes.
    """

    def __init__(self, output_dir: str = "./captured_io"):
        """
        Initialize I/O capture.

        Args:
            output_dir: Directory to store captured data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track captured data
        self._captured_requests: Dict[str, Any] = {}
        self._captured_outputs: Dict[str, Any] = {}
        self._captured_tasks: List[Dict] = []
        self._captured_worker_tasks: List[Dict] = []

        # Capture session info
        self._session_id = int(time.time())
        self._capture_enabled = True

        # Configuration snapshot
        self._config_snapshot: Optional[Dict] = None

    def enable(self):
        """Enable capture."""
        self._capture_enabled = True

    def disable(self):
        """Disable capture."""
        self._capture_enabled = False

    def is_enabled(self) -> bool:
        """Check if capture is enabled."""
        return self._capture_enabled

    def set_config(self, config: Any):
        """Capture configuration snapshot.

        Args:
            config: Configuration object (will be converted to dict)
        """
        if not self._capture_enabled:
            return

        self._config_snapshot = self._serialize_config(config)

    def capture_request(self, request: Any) -> Optional[str]:
        """Capture a request object.

        Args:
            request: Request object to capture

        Returns:
            Capture file path if captured, None otherwise
        """
        if not self._capture_enabled:
            return None

        request_id = getattr(request, "request_id", None)
        if request_id is None:
            request_id = f"req_{len(self._captured_requests)}"

        # Serialize request to dict
        data = self._serialize_request(request)

        self._captured_requests[request_id] = data

        # Save to file
        return self._save_data(
            data,
            IOTypes.REQUEST,
            request_id,
        )

    def capture_request_output(self, output: Any) -> Optional[str]:
        """Capture a request output object.

        Args:
            output: RequestOutput object to capture

        Returns:
            Capture file path if captured, None otherwise
        """
        if not self._capture_enabled:
            return None

        request_id = getattr(output, "request_id", None)
        if request_id is None:
            request_id = f"out_{len(self._captured_outputs)}"

        # Serialize output to dict
        data = self._serialize_request_output(output)

        self._captured_outputs[request_id] = data

        # Save to file
        return self._save_data(
            data,
            IOTypes.REQUEST_OUTPUT,
            request_id,
        )

    def capture_schedule_task(self, tasks: List[Any], current_id: int = -1) -> Optional[str]:
        """Capture tasks from scheduler.

        Args:
            tasks: List of task objects from scheduler
            current_id: Current scheduler ID

        Returns:
            Capture file path if captured, None otherwise
        """
        if not self._capture_enabled:
            return None

        task_data = {
            "current_id": current_id,
            "timestamp": time.time(),
            "num_tasks": len(tasks),
            "tasks": [],
        }

        for task in tasks:
            task_data["tasks"].append(self._serialize_task(task))

        self._captured_tasks.append(task_data)

        # Save to file
        filename = f"{IOTypes.SCHEDULE_TASK}_{self._session_id}_{len(self._captured_tasks)}.npz"
        filepath = self.output_dir / filename
        np.savez_compressed(filepath, data=pickle.dumps(task_data))

        return str(filepath)

    def capture_worker_task(self, tasks: List[Any], real_bsz: int) -> Optional[str]:
        """Capture tasks sent to worker.

        Args:
            tasks: List of task objects sent to worker
            real_bsz: Real batch size

        Returns:
            Capture file path if captured, None otherwise
        """
        if not self._capture_enabled:
            return None

        task_data = {
            "real_bsz": real_bsz,
            "timestamp": time.time(),
            "num_tasks": len(tasks),
            "tasks": [],
        }

        for task in tasks:
            task_data["tasks"].append(self._serialize_task(task))

        self._captured_worker_tasks.append(task_data)

        # Save to file
        filename = f"{IOTypes.WORKER_TASK}_{self._session_id}_{len(self._captured_worker_tasks)}.npz"
        filepath = self.output_dir / filename
        np.savez_compressed(filepath, data=pickle.dumps(task_data))

        return str(filepath)

    def save_config_snapshot(self) -> Optional[str]:
        """Save configuration snapshot.

        Returns:
            File path if saved, None otherwise
        """
        if not self._capture_enabled or self._config_snapshot is None:
            return None

        filename = f"config_{self._session_id}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._config_snapshot, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def save_index(self) -> Optional[str]:
        """Save index of all captured data.

        Returns:
            File path if saved, None otherwise
        """
        index = {
            "session_id": self._session_id,
            "timestamp": time.time(),
            "config_snapshot": self._save_config_snapshot(),
            "requests": list(self._captured_requests.keys()),
            "outputs": list(self._captured_outputs.keys()),
            "num_schedule_tasks": len(self._captured_tasks),
            "num_worker_tasks": len(self._captured_worker_tasks),
        }

        filename = f"index_{self._session_id}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def clear(self):
        """Clear all captured data."""
        self._captured_requests.clear()
        self._captured_outputs.clear()
        self._captured_tasks.clear()
        self._captured_worker_tasks.clear()
        self._config_snapshot = None

    def _serialize_request(self, request: Any) -> Dict:
        """Serialize request to dict for storage.

        Args:
            request: Request object

        Returns:
            Dictionary representation of request
        """
        data = {}

        # Get all important fields from request
        fields_to_capture = [
            "request_id",
            "prompt",
            "prompt_token_ids",
            "prompt_token_ids_len",
            "messages",
            "tools",
            "system",
            "history",
            "eos_token_ids",
            "sampling_params",
            "pooling_params",
            "multimodal_inputs",
            "multimodal_data",
            "disable_chat_template",
            "disaggregate_info",
            "draft_token_ids",
            "guided_json",
            "guided_regex",
            "guided_choice",
            "guided_grammar",
            "structural_tag",
            "guided_json_object",
            "image_position",
            "image_type_ids",
            "grid_thw",
            "chunk_index",
            "num_cached_tokens",
            "num_prefilled_chunks",
            "task_type",
            "idx",
            "user",
            "trace_carrier",
        ]

        for field in fields_to_capture:
            if hasattr(request, field):
                value = getattr(request, field)
                data[field] = self._serialize_value(value)

        # Also capture any additional attributes not in the predefined list
        # This ensures dynamic attributes like block_table are captured
        for attr_name in dir(request):
            if (
                not attr_name.startswith("_")
                and attr_name not in fields_to_capture
                and hasattr(request, attr_name)
                and not callable(getattr(request, attr_name))
            ):
                value = getattr(request, attr_name)
                data[attr_name] = self._serialize_value(value)

        return data

    def _serialize_request_output(self, output: Any) -> Dict:
        """Serialize request output to dict for storage.

        Args:
            output: RequestOutput object

        Returns:
            Dictionary representation of output
        """
        data = {}

        # Get all important fields from output
        fields_to_capture = [
            "request_id",
            "prompt",
            "prompt_token_ids",
            "prompt_logprobs",
            "outputs",
            "finished",
            "error_code",
            "error_msg",
            "num_cached_tokens",
            "num_input_image_tokens",
            "num_input_video_tokens",
            "encoder_prompt",
            "encoder_prompt_token_ids",
        ]

        for field in fields_to_capture:
            if hasattr(output, field):
                value = getattr(output, field)
                data[field] = self._serialize_value(value)

        return data

    def _serialize_task(self, task: Any) -> Dict:
        """Serialize task to dict for storage.

        Args:
            task: Task object

        Returns:
            Dictionary representation of task
        """
        data = {}

        fields_to_capture = [
            "request_id",
            "prompt_token_ids",
            "prompt_token_ids_len",
            "sampling_params",
            "block_table",
            "task_type",
            "idx",
            "user",
            "disaggregate_info",
            "num_cached_tokens",
            "num_prefilled_chunks",
            "metrics",
        ]

        for field in fields_to_capture:
            if hasattr(task, field):
                value = getattr(task, field)
                data[field] = self._serialize_value(value)

        return data

    def _serialize_config(self, config: Any) -> Dict:
        """Serialize configuration to dict.

        Args:
            config: Configuration object

        Returns:
            Dictionary representation of config
        """
        data = {}

        # Config sections that affect EngineService I/O
        config_sections = [
            "model_config",
            "scheduler_config",
            "cache_config",
            "parallel_config",
            "speculative_config",
            "structured_outputs_config",
            "eplb_config",
        ]

        for section in config_sections:
            if hasattr(config, section):
                section_obj = getattr(config, section)
                if hasattr(section_obj, "__dict__"):
                    data[section] = {}
                    for attr_name, attr_value in vars(section_obj).items():
                        data[section][attr_name] = self._serialize_value(attr_value)
                else:
                    data[section] = self._serialize_value(section_obj)

        return data

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for storage.

        Args:
            value: Value to serialize

        Returns:
            Serialized value (dict, list, or primitive)
        """
        if value is None:
            return None
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, np.ndarray):
            return {
                "_type": "numpy.ndarray",
                "shape": value.shape,
                "dtype": str(value.dtype),
                "data": value.tolist(),
            }
        elif hasattr(value, "__dict__"):
            # Convert dataclass or object to dict
            return {k: self._serialize_value(v) for k, v in vars(value).items()}
        else:
            # For enums and other types
            return str(value)

    def _save_data(self, data: Dict, io_type: str, request_id: str) -> str:
        """Save data to file.

        Args:
            data: Data to save
            io_type: Type of I/O data
            request_id: Request ID

        Returns:
            File path
        """
        # Use pickle for Python objects, JSON for compatibility
        filename = f"{io_type}_{request_id}.npz"
        filepath = self.output_dir / filename
        np.savez_compressed(filepath, data=pickle.dumps(data))

        return str(filepath)

    def _save_config_snapshot(self) -> Optional[str]:
        """Save config snapshot as JSON."""
        if self._config_snapshot is None:
            return None

        filename = f"config_{self._session_id}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._config_snapshot, f, indent=2, ensure_ascii=False)

        return str(filepath)


# Global capture instance
_global_capture: Optional[EngineIOCapture] = None


def get_global_capture() -> EngineIOCapture:
    """Get or create global capture instance."""
    global _global_capture
    if _global_capture is None:
        _global_capture = EngineIOCapture()
    return _global_capture


def enable_capture(output_dir: str = "./captured_io"):
    """Enable global capture with specified output directory.

    Args:
        output_dir: Directory to store captured data
    """
    global _global_capture
    _global_capture = EngineIOCapture(output_dir)
    _global_capture.enable()


def disable_capture():
    """Disable global capture."""
    global _global_capture
    if _global_capture is not None:
        _global_capture.disable()


def is_capture_enabled() -> bool:
    """Check if global capture is enabled."""
    global _global_capture
    return _global_capture is not None and _global_capture.is_enabled()


class IOCaptureDecorator:
    """Decorator for capturing I/O in methods."""

    def __init__(self, capture_type: str):
        """
        Initialize decorator.

        Args:
            capture_type: Type of I/O to capture
        """
        self.capture_type = capture_type

    def __call__(self, func):
        """Wrap function to capture I/O."""

        def wrapper(*args, **kwargs):
            if not is_capture_enabled():
                return func(*args, **kwargs)

            capture = get_global_capture()

            # Capture input
            if self.capture_type == IOTypes.REQUEST and args:
                # First arg is typically self, second might be request
                if len(args) > 1:
                    capture.capture_request(args[1])

            # Call function
            result = func(*args, **kwargs)

            # Capture output
            if self.capture_type == IOTypes.REQUEST_OUTPUT and result is not None:
                if isinstance(result, list) and len(result) > 0:
                    for item in result:
                        capture.capture_request_output(item)
                else:
                    capture.capture_request_output(result)

            return result

        return wrapper
