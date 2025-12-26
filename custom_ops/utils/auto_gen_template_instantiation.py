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
"""Universal template instantiation generator - fully based on configuration file template instantiation generation."""

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TemplateConfig:
    """Template configuration class."""

    name: str  # Function name
    function_name: str  # Actual function name
    impl_file: str  # Implementation file path
    template_params: List[str]  # Template parameter list (in order)
    dispatch_params: Dict[str, List[Any]]  # Dispatch parameters
    data_types: Optional[List[Tuple[str, str, str]]] = None  # Data type combinations (input_type, output_type, suffix)
    max_instances_per_file: int = 60  # Maximum instances per file
    file_prefix: str = ""  # File prefix
    function_signature: str = ""  # Function signature template


class UniversalTemplateInstantiator:
    """Universal template instantiator - fully based on configuration file."""

    def __init__(self, config_file: str):
        """Initialize the instantiator."""
        self.config_file = config_file
        self.configs = self._load_configs()

    def _load_configs(self) -> Dict[str, TemplateConfig]:
        """Load configuration file."""
        with open(self.config_file, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        configs = {}
        for name, config_dict in config_data.items():
            config = TemplateConfig(**config_dict)
            self._validate_config(config)
            configs[name] = config
        return configs

    def _validate_config(self, config: TemplateConfig):
        """Validate configuration completeness."""
        has_t = "T" in config.template_params
        has_out_t = "OutT" in config.template_params

        if (has_t or has_out_t) and not config.data_types:
            raise ValueError(
                f"Configuration '{config.name}' has T or OutT in template_params but no data_types configured"
            )

        # Skip validation for special handled functions
        if config.name == "moe_fast_hardamard_impl":
            return

        special_params = {"T", "OutT", "NUM_WARP_Q"}
        for param_name in config.template_params:
            if param_name not in special_params and param_name not in config.dispatch_params:
                raise ValueError(f"Template parameter '{param_name}' in '{config.name}' not found in dispatch_params")

        if "NUM_WARP_Q" in config.template_params and "BLOCK_SHAPE_Q" not in config.dispatch_params:
            raise ValueError(
                f"Template parameter 'NUM_WARP_Q' in '{config.name}' requires 'BLOCK_SHAPE_Q' in dispatch_params"
            )

    def _calculate_num_warp_q(self, block_shape_q: int) -> int:
        """Calculate number of warps."""
        if block_shape_q <= 32:
            return 1
        else:
            return 4

    def _build_template_args(self, config: TemplateConfig, t_in: str, t_out: str, params: Dict[str, Any]) -> str:
        """Build template arguments."""
        template_args_parts = []

        for param_name in config.template_params:
            if param_name == "T":
                if t_in:
                    template_args_parts.append(t_in)
                else:
                    raise ValueError("Template parameter 'T' requires input type, but data_types is empty or invalid")
            elif param_name == "OutT":
                if t_out:
                    template_args_parts.append(t_out)
                else:
                    raise ValueError(
                        "Template parameter 'OutT' requires output type, but data_types is empty or invalid"
                    )
            elif param_name == "NUM_WARP_Q":
                if "BLOCK_SHAPE_Q" in params:
                    num_warp_q = self._calculate_num_warp_q(params["BLOCK_SHAPE_Q"])
                    template_args_parts.append(str(num_warp_q))
                else:
                    raise ValueError("Template parameter 'NUM_WARP_Q' requires 'BLOCK_SHAPE_Q' in dispatch_params")
            elif param_name in params:
                template_args_parts.append(str(params[param_name]))
            else:
                raise ValueError(f"Template parameter '{param_name}' not found in dispatch_params")

        return f"<{', '.join(template_args_parts)}>"

    def _generate_function_signature(
        self, config: TemplateConfig, template_args: str, t_in: str = "", t_out: str = ""
    ) -> str:
        """Generate function signature."""
        if config.function_signature:
            signature = config.function_signature.format(
                function_name=config.function_name, template_args=template_args
            )
            # Replace T and OutT with actual types if provided
            if t_in:
                signature = signature.replace("const T *", f"const {t_in} *")
            if t_out:
                signature = signature.replace("OutT*", f"{t_out}*")
            return signature
        else:
            raise ValueError(f"Function signature not found for {config.name}")

    def _generate_file_header(self, config: TemplateConfig) -> str:
        """Generate file header."""
        return f"""// Generated by autogen_template_instantiation.py - Do not edit.

#pragma once

#include "../../{config.impl_file}"
"""

    def _generate_template_instantiation(
        self, config: TemplateConfig, t_in: str, t_out: str, params: Dict[str, Any]
    ) -> str:
        """Generate template instantiation."""
        template_args = self._build_template_args(config, t_in, t_out, params)
        return self._generate_function_signature(config, template_args, t_in, t_out)

    def _clean_output_directory(self, output_dir: str):
        """Clean output directory before generating new files."""
        output_path = Path(output_dir)
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    def generate_combinations_for_type(self, config: TemplateConfig, t_in: str, t_out: str) -> List[Dict[str, Any]]:
        """Generate parameter combinations for specific type."""
        combinations = []

        if config.name == "moe_fast_hardamard_impl":
            combinations = self._generate_moe_hardamard_combinations(config, t_in, t_out)
        else:

            def _generate_recursive(
                params_dict: Dict[str, List[Any]], current_params: Dict[str, Any], param_names: List[str]
            ):
                if not param_names:
                    combinations.append(current_params.copy())
                    return

                param_name = param_names[0]
                for value in params_dict[param_name]:
                    current_params[param_name] = value
                    _generate_recursive(params_dict, current_params, param_names[1:])

            _generate_recursive(config.dispatch_params, {}, list(config.dispatch_params.keys()))

        return combinations

    def _generate_moe_hardamard_combinations(
        self, config: TemplateConfig, t_in: str, t_out: str
    ) -> List[Dict[str, Any]]:
        """Generate combinations for MoeFastHardamardImplWrapper based on code logic."""
        combinations = []

        for vec_size in [1, 2, 4, 8, 16]:
            for log_n in [7, 8, 9, 10]:
                combinations.append(
                    {"kLogN": log_n, "VecSize": vec_size, "kNChunks": 1, "kThreads": 128, "UseDiagonalBlockMatrix": 1}
                )

        for log_n in [7, 8, 9, 10]:
            vec_size = (1 << log_n) // 128
            combinations.append(
                {"kLogN": log_n, "VecSize": vec_size, "kNChunks": 28, "kThreads": 128, "UseDiagonalBlockMatrix": 0}
            )
            combinations.append(
                {"kLogN": log_n, "VecSize": vec_size, "kNChunks": 36, "kThreads": 128, "UseDiagonalBlockMatrix": 0}
            )

        for log_n in [11, 12, 13, 14]:
            vec_size = 8
            n_chunks = (1 << log_n) // (128 * vec_size)
            combinations.append(
                {
                    "kLogN": log_n,
                    "VecSize": vec_size,
                    "kNChunks": n_chunks,
                    "kThreads": 128,
                    "UseDiagonalBlockMatrix": 0,
                }
            )

        return combinations

    def split_combinations(self, combinations: List[Dict[str, Any]], max_per_file: int) -> List[List[Dict[str, Any]]]:
        """Split combinations into multiple files."""
        chunks = []
        for i in range(0, len(combinations), max_per_file):
            chunk = combinations[i : i + max_per_file]
            chunks.append(chunk)
        return chunks

    def generate_file_content(
        self,
        config: TemplateConfig,
        t_in: str,
        t_out: str,
        t_out_name: str,
        file_index: int,
        combinations: List[Dict[str, Any]],
    ) -> str:
        """Generate file content."""
        content = self._generate_file_header(config)

        for params in combinations:
            content += self._generate_template_instantiation(config, t_in, t_out, params)

        return content

    def generate_for_function_type(self, function_name: str, output_dir: str):
        """Generate template instantiation files for specific function type."""
        if function_name not in self.configs:
            raise ValueError(f"Function type '{function_name}' not found in config")

        config = self.configs[function_name]
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not config.data_types:
            data_types = [("", "", "")]
        else:
            data_types = config.data_types

        for t_in, t_out, t_out_name in data_types:
            combinations = self.generate_combinations_for_type(config, t_in, t_out)
            if combinations:
                chunks = self.split_combinations(combinations, config.max_instances_per_file)
                for i, chunk in enumerate(chunks):
                    filename = f"{config.file_prefix}{t_out_name}_part_{i:02d}.cu"
                    filepath = output_path / filename
                    content = self.generate_file_content(config, t_in, t_out, t_out_name, i, chunk)
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(content)

    def generate_all(self, output_dir: str):
        """Generate all configured function types."""
        self._clean_output_directory(output_dir)
        for function_name in self.configs.keys():
            print(f"Generating template instantiations for {function_name}...")
            self.generate_for_function_type(function_name, output_dir)
            print(f"Completed generating {function_name} template instantiations.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Universal template instantiation generator")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Configuration file path (JSON format)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory",
    )

    args = parser.parse_args()

    try:
        instantiator = UniversalTemplateInstantiator(args.config)
        instantiator.generate_all(args.output)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
