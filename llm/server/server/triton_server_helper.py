# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import queue
import socket
import subprocess
import time
from collections import defaultdict
from multiprocessing import shared_memory

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from server.engine.config import Config
from server.utils import get_logger

app = FastAPI()

logger = get_logger("health_checker", "health_checker.log")
env_config = Config()

@app.get("/v2/health/ready")
def check_health():
    """
    探活接口"""
    status, error_info = check()
    if status is True:
        logger.info("check_health: OK")
        return Response()
    else:
        logger.info("check_health: Bad")
        return JSONResponse(
                status_code=500,
                content=error_info)


@app.get("/v2/health/live")
def check_live():
    """
    探活接口"""
    status, error_info = check()
    if status is True:
        logger.info("check_health: OK")
        return Response()
    else:
        logger.info("check_health: Bad")
        return JSONResponse(
                status_code=500,
                content=error_info)


def check_infer_engine_process():
    # 检查infer进程是否存在
    mp_num = int(env_config.mp_num)
    for i in range(mp_num):
        try:
            infer_live_flag_shm = shared_memory.SharedMemory(name=env_config.get_unique_name("shm_flag_infer_{}_live".format(i)))
        except Exception as e:  # infer掉了会报异常
            return False
    return True


def check():
    """
    推理服务的状态探活接口
    """
    error_info = {}
    grpc_port = os.getenv("GRPC_PORT")

    # 1. 检查server是否健康
    if grpc_port is not None:
        sock = socket.socket()
        try:
            sock.connect(('localhost', int(grpc_port)))
        except Exception:
            error_info["error_code"] = 1
            error_info["error_msg"] = "server is not ready"
            logger.info("server is not ready")
            return False, error_info
        finally:
            sock.close()

    # 2. 检查engine是否健康
    is_engine_live = check_infer_engine_process()
    if is_engine_live is False:
        error_info["error_code"] = 2
        error_info["error_msg"] = "infer engine is down"
        logger.info("infer engine is down")
        return False, error_info

    # 检查是否启动
    engine_ready_checker = np.ndarray(engine_ready_check_flag.shape, dtype=engine_ready_check_flag.dtype,
                                      buffer=shm_engine_ready_check_flag.buf)
    if engine_ready_checker[0] == 0:  # 值为0代表没启动，值为1代表已启动
        error_info["error_code"] = 2
        error_info["error_msg"] = "infer engine is down"
        logger.info("infer engine is down")
        return False, error_info

    # 检查是否hang住
    engine_hang_checker = np.ndarray(engine_healthy_recorded_time.shape, dtype=engine_healthy_recorded_time.dtype,
                                buffer=shm_engine_healthy_recorded_time.buf)
    elapsed_time = time.time() - engine_hang_checker[0]
    logger.info("engine_checker elapsed time: {}".format(elapsed_time))
    if (engine_hang_checker[0]) and (elapsed_time > time_interval_threashold):
        error_info["error_code"] = 3
        error_info["error_msg"] = "infer engine hangs"
        logger.info("infer engine hangs")
        return False, error_info

    return True, error_info


def start_health_checker(http_port):
    import sys
    sys.stdout = open("log/health_http.log", 'w')
    sys.stderr = sys.stdout
    uvicorn.run(app=app, host='0.0.0.0', port=http_port, workers=1, log_level="info")


time_interval_threashold = env_config.check_health_interval    # 10s infer engine没有执行过while循环，则判定hang死或挂掉等问题
engine_healthy_recorded_time = np.zeros([1], dtype=float)
shm_engine_healthy_recorded_time = shared_memory.SharedMemory(
        create=True,
        size=engine_healthy_recorded_time.nbytes,
        name=env_config.get_unique_name("engine_healthy_recorded_time"))  # 由推理引擎进行更新，每次读token时候就刷新一次时间，正常情况下该时间戳在30s内肯定会被刷新

engine_ready_check_flag = np.zeros([1], dtype=np.int32)
shm_engine_ready_check_flag = shared_memory.SharedMemory(
        create=True,
        size=engine_ready_check_flag.nbytes,
        name=env_config.get_unique_name("engine_ready_check_flag"))    # 由推理引擎更新，推理引擎初始化完毕时候置为1
