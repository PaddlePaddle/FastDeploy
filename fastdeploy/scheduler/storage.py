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

import re
from collections.abc import Awaitable
from typing import List, Optional, Union

import redis
from packaging import version
from redis.typing import EncodableT, FieldT, KeyT, Number, ResponseT

LUA_LPOP = """
local key = KEYS[1]
local count = tonumber(ARGV[1])
local elements = redis.call('LRANGE', key, 0, count - 1)
local elementsCount = #elements
if elementsCount > 0 then
    redis.call('LTRIM', key, count, -1)
end
return elements
"""

LUA_ZINCRBY = """
local key = KEYS[1]
local amount = tonumber(ARGV[1])
local value = ARGV[2]
local rem_amount = ARGV[5] == 'NIL' and nil or tonumber(ARGV[3])

local currentAmount = redis.call('ZINCRBY',key, amount, value)
currentAmount = tonumber(currentAmount) or 0

if rem_amount ~= nil and currentAmount == rem_amount then
    redis.call('ZREM', key, value)
    currentAmount = 0
end

return currentAmount
"""


class AdaptedRedis(redis.Redis):
    """
    A Redis client adapter that provides version-compatible operations.

    This class extends the standard Redis client to:
    - Handle version-specific behavior differences
    - Add TTL support for list operations
    - Provide atomic operations with expiration
    - Implement custom Lua scripts for enhanced functionality
    """

    def __init__(self, **kwargs):
        """
        Initialize the AdaptedRedis client.

        Args:
            **kwargs: Standard Redis client connection parameters
        """
        super().__init__(**kwargs)

        self._old_version = False
        self._parse_version()
        self._register_script()

    def _parse_version(self):
        """
        Parse and store the Redis server version.

        Determines if the server is an older version that requires
        special handling for certain operations.
        """
        server_info = self.info(section="server")
        version_string = server_info["redis_version"]

        match = re.search(r"^(\d+\.\d+\.\d+)", version_string)
        if match:
            redis_version = match.group(1)
        else:
            redis_version = "0.0.0"

        current_version = version.parse(redis_version)
        target_version = version.parse("6.2.28")

        if current_version <= target_version:
            self._old_version = True

        self.version = redis_version

    def _register_script(self):
        """
        Register custom Lua scripts for enhanced Redis operations.

        Scripts include:
        - Atomic LPOP with count (for older Redis versions)
        - ZINCRBY with removal threshold
        """
        if self._old_version:
            self._lpop = self.register_script(LUA_LPOP)
        self._zincrby = self.register_script(LUA_ZINCRBY)

    def rpush(self, name: str, *values: FieldT, ttl: Optional[float] = None) -> Union[Awaitable[int], int]:
        """
        RPUSH operation with optional TTL.

        Args:
            name: List key
            *values: Values to push
            ttl: Optional time-to-live in seconds

        Returns:
            Length of the list after push
        """
        if ttl is None:
            return super().rpush(name, *values)

        with self.pipeline() as pipe:
            pipe.multi()
            pipe.rpush(name, *values)
            pipe.expire(name, ttl)
            result = pipe.execute()
            return result[0]

    def zincrby(
        self,
        name: KeyT,
        amount: float,
        value: EncodableT,
        rem_amount: Optional[float] = None,
        ttl: Optional[float] = None,
    ) -> ResponseT:
        """
        Atomic ZINCRBY with removal threshold and optional TTL.

        Args:
            name: Sorted set key
            amount: Increment amount
            value: Member to increment
            rem_amount: Optional threshold for member removal
            ttl: Optional time-to-live in seconds

        Returns:
            New score of the member
        """
        amount = str(amount)

        if ttl is None:
            if rem_amount is None:
                return super().zincrby(name, amount, value)
            rem_amount = "NIL" if rem_amount is None else str(rem_amount)
            return self._zincrby(keys=[name], args=[amount, value, rem_amount])

        with self.pipeline() as pipe:
            pipe.multi()
            if rem_amount is None:
                pipe.zincrby(name, amount, value)
            else:
                rem_amount = "NIL" if rem_amount is None else str(rem_amount)
                self._zincrby(keys=[name], args=[amount, value, rem_amount], client=pipe)
            pipe.expire(name, ttl)
            result = pipe.execute()
            return result[0]

    def lpop(
        self,
        name: str,
        count: Optional[int] = None,
        ttl: Optional[float] = None,
    ) -> Union[Awaitable[Union[str, List, None]], Union[str, List, None]]:
        """
        LPOP operation with count support and optional TTL.

        Args:
            name: List key
            count: Number of elements to pop
            ttl: Optional time-to-live in seconds

        Returns:
            Popped elements (single or list)
        """
        if ttl is None:
            if self._old_version and count is not None:
                return self._lpop(keys=[name], args=[count])
            return super().lpop(name, count)

        with self.pipeline() as pipe:
            pipe.multi()
            if self._old_version and count is not None:
                self._lpop(keys=[name], args=[count], client=pipe)
            else:
                pipe.lpop(name, count)
            pipe.expire(name, ttl)
            result = pipe.execute()
            return result[0]

    def blpop(self, keys: List, timeout: Optional[Number] = 0):
        """
        BLPOP operation with version-specific timeout handling.

        Args:
            keys: List of keys to pop from
            timeout: Maximum wait time in seconds

        Returns:
            Tuple of (key, value) or None if timeout
        """
        if self._old_version:
            if timeout > 0 and timeout < 1:
                timeout = 1
            timeout = int(timeout)
            return super().blpop(keys=keys, timeout=timeout)

        if timeout > 0 and timeout < 0.01:
            timeout = 0.01
        return super().blpop(keys=keys, timeout=timeout)
