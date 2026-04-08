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

from fastdeploy.router.router import RouterArgs, launch_router
from fastdeploy.utils import FlexibleArgumentParser
from fastdeploy.utils import router_logger as logger


def main() -> None:
    parser = FlexibleArgumentParser()
    parser = RouterArgs.add_cli_args(parser)
    args = parser.parse_args()

    try:
        launch_router(args)
    except Exception as e:
        logger.error(f"Error starting router: {e}")
        raise e


if __name__ == "__main__":
    main()
