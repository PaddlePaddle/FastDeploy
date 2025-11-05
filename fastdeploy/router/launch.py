"""
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

import argparse

from fastdeploy.router.router import start_router
from fastdeploy.utils import router_logger as logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Router for splitwise deployment testing")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the router server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default="9000",
        help="Port number to bind the router server",
    )
    parser.add_argument(
        "--splitwise",
        action="store_true",
        help="Router uses splitwise deployment",
    )
    parser.add_argument(
        "--request-timeout-secs",
        type=int,
        default=1800,
        help="Request timeout in seconds",
    )
    args = parser.parse_args()

    try:
        start_router(args)
    except Exception as e:
        logger.error(f"Error starting router: {e}")
        raise e


if __name__ == "__main__":
    main()
