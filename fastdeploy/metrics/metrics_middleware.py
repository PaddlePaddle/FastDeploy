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

"""
metrics middleware
"""

import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from fastdeploy.metrics.metrics import main_process_metrics

EXCLUDE_PATHS = ["/health", "/metrics", "/ping", "/load", "/v1/chat/completions", "/v1/completions"]


# --- process http metrics ---
class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):

        # exclude some paths
        if request.url.path in EXCLUDE_PATHS:
            return await call_next(request)

        start_time = time.time()
        status_code = 500
        path = request.url.path
        method = request.method

        try:
            # call next
            response = await call_next(request)
            status_code = response.status_code

        finally:
            end_time = time.time()
            # calculate time
            process_time = end_time - start_time

            # record http metrics
            main_process_metrics.http_requests_total.labels(method=method, path=path, status_code=status_code).inc()
            main_process_metrics.http_request_duration_seconds.labels(method=method, path=path).observe(process_time)

        return response
