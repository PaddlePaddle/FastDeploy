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


from typing import Optional, List
from redis.typing import Number
import redis
from packaging import version
import re


LUA_SCRIPT_LPOP = """
local key = KEYS[1]
local count = tonumber(ARGV[1])
local elements = redis.call('LRANGE', key, 0, count - 1)
local elementsCount = #elements
if elementsCount > 0 then
    redis.call('LTRIM', key, count, -1)
end
return elements
"""

class AdaptedRedis(redis.Redis):
    """
        AdaptedRedis class: Adapt to different versions of Redis
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._old_version = False
        self._parse_version()
        self._warm_up()

    def _parse_version(self):
        """
            parse version
        """
        server_info = self.info(section='server')
        version_string = server_info['redis_version']

        match = re.search(r'^(\d+\.\d+\.\d+)', version_string)
        if match:
            redis_version = match.group(1)
        else:
            redis_version = "0.0.0"

        current_version = version.parse(redis_version)
        target_version = version.parse("6.2.28")

        if current_version <= target_version:
            self._old_version = True

        self.version = redis_version

    def _warm_up(self):
        """
            preload some lua scripts
        """
        if self._old_version:
            self._lpop = self.register_script(LUA_SCRIPT_LPOP)

    def lpop(self, name: str, count: Optional[int] = None):
        """
            similar to redis lpop
        """
        if self._old_version and count is not None:
            return self._lpop(keys=[name], args=[count])
        return super().lpop(name, count)

    def blpop(self, keys: List, timeout: Optional[Number] = 0):
        """
            similar to redis blpop
        """
        if self._old_version:
            if timeout > 0 and timeout < 1:
                timeout = 1
            timeout = int(timeout)
            return super().blpop(keys=keys, timeout=timeout)

        if timeout > 0 and timeout < 0.01:
            timeout = 0.01
        return super().blpop(keys=keys, timeout=timeout)