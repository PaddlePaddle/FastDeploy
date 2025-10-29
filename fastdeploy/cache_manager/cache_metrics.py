"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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

from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.utils import get_logger

logger = get_logger("prefix_cache_manager", "cache_manager.log")


class CacheMetrics:
    """
    Cache Metrics used to record the cache hit time, token num, request num, etc.
    """

    def __init__(self):
        self.total_match_time = 0.0
        self.avg_match_time = 0.0
        self.min_match_time = 1e9
        self.max_match_time = 0.0

        # request level
        self.req_count = 0
        self.hit_req_count = 0
        self.hit_req_ratio = 0.0

        # token level
        self.total_gpu_matched_token_num = 0
        self.total_cpu_matched_token_num = 0

        self.matched_token_num = 0
        self.total_token_num = 0
        self.hit_token_ratio = 0.0
        self.cpu_hit_token_ratio = 0.0
        self.gpu_hit_token_ratio = 0.0

    def _update_history_hit_metrics(self):
        """
        update hit ratio
        """
        self.hit_req_ratio = self.hit_req_count / self.req_count
        self.hit_token_ratio = self.matched_token_num / self.total_token_num
        self.cpu_hit_token_ratio = self.total_cpu_matched_token_num / self.total_token_num
        self.gpu_hit_token_ratio = self.total_gpu_matched_token_num / self.total_token_num

        main_process_metrics.hit_req_rate.set(self.hit_req_ratio)
        main_process_metrics.hit_token_rate.set(self.hit_token_ratio)
        main_process_metrics.cpu_hit_token_rate.set(self.cpu_hit_token_ratio)
        main_process_metrics.gpu_hit_token_rate.set(self.gpu_hit_token_ratio)

        logger.info(
            f"Metrics for all requests: req_count {self.req_count} hit_req_count {self.hit_req_count}"
            + f" hit_req_ratio {self.hit_req_ratio:.2f} hit_token_ratio {self.hit_token_ratio:.2f}"
            + f" gpu_hit_token_ratio {self.gpu_hit_token_ratio:.2f}"
            + f" cpu_hit_token_ratio {self.cpu_hit_token_ratio:.2f}"
            + f" total_gpu_matched_token_num {self.total_gpu_matched_token_num}"
            + f" total_cpu_matched_token_num {self.total_cpu_matched_token_num}"
            + f" total_matched_token_num {self.matched_token_num}"
            + f" total_token_num {self.total_token_num}"
        )

    def calculate_hit_metrics(
        self,
        req_id,
        current_query_cpu_match_token_num,
        current_query_gpu_match_token_num,
        current_query_token_num,
    ):
        """
        calculate hit metrics for current query
        """

        cpu_cache_match_ratio = current_query_cpu_match_token_num / current_query_token_num
        gpu_cache_match_ratio = current_query_gpu_match_token_num / current_query_token_num

        total_match_ratio = cpu_cache_match_ratio + gpu_cache_match_ratio

        self.total_cpu_matched_token_num += current_query_cpu_match_token_num
        self.total_gpu_matched_token_num += current_query_gpu_match_token_num

        self.matched_token_num += current_query_cpu_match_token_num + current_query_gpu_match_token_num
        self.total_token_num += current_query_token_num
        logger.info(
            f"Metrics for req_id {req_id}: token_num {current_query_token_num}"
            + f" cpu_cache_match_ratio {cpu_cache_match_ratio}"
            + f" gpu_cache_match_ratio {gpu_cache_match_ratio}"
            + f" total_match_ratio {total_match_ratio}"
        )

    def reset_metrics(self):
        """
        reset metrics
        """
        self.total_match_time = 0.0
        self.avg_match_time = 0.0
        self.min_match_time = 1e9
        self.max_match_time = 0.0

        self.req_count = 0
        self.hit_req_count = 0
        self.hit_req_ratio = 0.0

        self.total_gpu_matched_token_num = 0
        self.total_cpu_matched_token_num = 0

        self.matched_token_num = 0
        self.total_token_num = 0
        self.hit_token_ratio = 0.0
        self.cpu_hit_token_ratio = 0.0
        self.gpu_hit_token_ratio = 0.0
