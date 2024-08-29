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

import os
import threading
import time
from queue import Queue
from multiprocessing.managers import (
    AcquirerProxy,
    BaseManager,
    ListProxy,
    Value,
    ValueProxy,
)

from server.utils import get_logger

logger = get_logger("infer_server", "task_queue_manager.log")


class QueueManager(BaseManager):
    """
    基础类
    """

    pass


class TaskQueueManager(object):
    """
    管理类
    """

    def __init__(self, rank=0, mp_num=8, port=56666):
        """
        初始化函数，用于创建对象时进行初始化操作。
        """
        self.max_get_num = int(os.getenv("ENGINE_MAX_NEED_NUM", 0))
        QueueManager.register('get_list')
        QueueManager.register('get_value')
        QueueManager.register('get_lock')
        QueueManager.register('get_barrier1')
        QueueManager.register('get_barrier2')
        QueueManager.register('get_queue')

        self.client_manager = QueueManager(address=('127.0.0.1', port),
                                           authkey=b'infer_queue'
                                           )
        self.client_manager.connect()
        self.list = self.client_manager.get_list()
        self.value = self.client_manager.get_value()
        self.lock = self.client_manager.get_lock()
        self.barrier1 = self.client_manager.get_barrier1()
        self.barrier2 = self.client_manager.get_barrier2()
        self.queue = self.client_manager.get_queue()
        self.mp_num = mp_num
        self.rank = rank
        self.position = 1 << rank
        self.total_num = (1 << self.mp_num) - 1
        logger.info(f"init task queue manager success, rank: {rank}")

    def empty(self):
        """
        暴露至推理端，用于判断队列是否为空
        """
        try:
            return len(self.list) == 0
        except Exception as e:
            logger.error(f"empty function meets error: {e}")
            raise e

    def put(self, item):
        """
        向队列中添加数据
        """
        self.lock.acquire()
        if 0 < self.value.get() < self.total_num:
            self.lock.release()
            while 0 < self.value.get() < self.total_num:
                time.sleep(0.001)
            logger.info("put item to queue wait finish")
            self.lock.acquire()
        if self.max_get_num <= 0 and self.value.get() == self.total_num:
            self.list[:] = []
        self.value.set(0)
        self.list.append(item)
        self.lock.release()
        logger.info("put item to queue success")

    def get(self):
        """
        从队列中获取数据
        """
        input_list = []
        read_finish = False
        self.lock.acquire()
        if self.value.get() & self.position == 0 and len(self.list) > 0:
            # 控制进入引擎的输入数量. 默认服务中所有输入都拷贝进引擎一起处理
            if self.max_get_num > 0:
                input_list.extend(self.list[: self.max_get_num])
            else:
                input_list.extend(self.list[:])
            set_value = self.value.get() | self.position
            logger.info("rank: {0} set_value: {1}".format(self.rank, set_value))
            if set_value >= self.total_num:
                if self.max_get_num > 0:
                    for i in range(self.max_get_num):
                        self.list.pop(0)
                else:
                    self.list[:] = []
                set_value = 0
                read_finish = True
            self.value.set(set_value)
        self.lock.release()
        return input_list, read_finish


def launch_queue_service(port, num_workers):
    """
    启动进程间通信队列服务

    port: 监听端口号
    num_workers: infer进程的数量
    """
    try:
        logger.info(f"start launch queue service, port:{port}")
        value = Value("i", 0)
        QueueManager.register("get_value", callable=lambda: value, proxytype=ValueProxy)
        List = list()
        QueueManager.register("get_list", callable=lambda: List, proxytype=ListProxy)
        lock = threading.Lock()
        QueueManager.register('get_lock',
                            callable=lambda: lock,
                            proxytype=AcquirerProxy)
        barrier1 = threading.Barrier(num_workers)
        QueueManager.register('get_barrier1', callable=lambda: barrier1)
        barrier2 = threading.Barrier(num_workers)
        QueueManager.register('get_barrier2', callable=lambda: barrier2)
        q = Queue()
        QueueManager.register("get_queue", callable=lambda: q)
        m = QueueManager(address=('127.0.0.1', port), authkey=b'infer_queue')
        s = m.get_server()
        logger.info("launch queue service success")
        s.serve_forever()
        logger.info("finish queue service")
    except Exception as e:
        logger.error(f"launch queue service failed, error_msg: {e}")
        raise e
