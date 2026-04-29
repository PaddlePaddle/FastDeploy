"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from typing import TYPE_CHECKING

from .base import Proposer

if TYPE_CHECKING:
    from fastdeploy.config import FDConfig


class NaiveProposer(Proposer):
    """
    Proposer for NaiveProposer.

    Not propose draft tokens, simply utilizing the framework
    to place the last autoregressively generated token in
    the first position of draft_tokens.
    """

    def __init__(self, fd_config: "FDConfig"):
        super().__init__(fd_config)

    def _run_impl(self):
        return
