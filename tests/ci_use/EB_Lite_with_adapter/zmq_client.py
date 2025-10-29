import threading
import time
import uuid
from threading import Event

import msgpack
import zmq


class LLMReqClient:
    """
    LLM request client
    """

    def __init__(self, ip, send_req_server_port, recv_res_server_port):
        self.ZMQ_SNDHWM = 64 * 1024
        self.context = zmq.Context()
        self.send_req_client = self.context.socket(zmq.PUSH)
        self.recv_res_client = self.context.socket(zmq.DEALER)
        self.send_req_client.setsockopt(zmq.SNDHWM, self.ZMQ_SNDHWM)
        self.send_req_client.setsockopt(zmq.SNDTIMEO, -1)
        self.recv_res_client.setsockopt(zmq.SNDHWM, self.ZMQ_SNDHWM)
        self.recv_res_client.setsockopt(zmq.SNDTIMEO, -1)
        self.send_req_client.connect(f"tcp://{ip}:{send_req_server_port}")
        self.recv_res_client.connect(f"tcp://{ip}:{recv_res_server_port}")
        self.need_exit = False
        self.response_socket_lock = threading.Lock()

    def send_request(self, req_data):
        self.send_req_client.send_json(req_data)

    def request_result(self, req_id):
        with self.response_socket_lock:
            print(f"request result data for {req_id}")
            self.recv_res_client.send_multipart([b"", req_id.encode("utf-8")])

    def consume_results(self, result_queue):
        while True:
            try:
                try:
                    with self.response_socket_lock:
                        frames = self.recv_res_client.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    time.sleep(0.001)
                    continue
                data = frames[-1]
                response = msgpack.unpackb(data)
                # print(f"get result data {response}")
                result_queue.put(response)
                if self.need_exit:
                    break
            except Exception as e:
                print(f"zmq client occurred error {e} type: {type(e)} frames: {frames}")

    def start(self, result_queue):
        threading.Thread(target=self.consume_results, args=(result_queue,), daemon=True).start()

    def exit(self):
        print("exit")
        self.need_exit = True


class LLMControlClient:
    """
    LLM control client
    """

    def __init__(self, ip, port):
        self.ZMQ_SNDHWM = 64 * 1024
        self.context = zmq.Context()
        self.control_client = self.context.socket(zmq.DEALER)
        self.control_client.setsockopt(zmq.SNDHWM, self.ZMQ_SNDHWM)
        self.control_client.setsockopt(zmq.SNDTIMEO, -1)
        self.control_client.connect(f"tcp://{ip}:{port}")
        self.task_event = {}
        self.result = {}
        self.response_socket_lock = threading.Lock()
        threading.Thread(target=self.recv_results, daemon=True).start()

    def get_payload(self):
        task_id = f"get_payload_{uuid.uuid4()}"
        task = {"task_id": task_id, "cmd": "get_payload"}
        self.task_event[task_id] = Event()
        payload = msgpack.packb(task)
        with self.response_socket_lock:
            self.control_client.send_multipart([b"", payload])
        self.task_event[task_id].wait()
        result = self.result[task_id]
        del self.result[task_id]
        del self.task_event[task_id]
        return result

    def get_metrics(self):
        task_id = f"get_metrics_{uuid.uuid4()}"
        task = {"task_id": task_id, "cmd": "get_metrics"}
        self.task_event[task_id] = Event()
        payload = msgpack.packb(task)
        with self.response_socket_lock:
            self.control_client.send_multipart([b"", payload])
        self.task_event[task_id].wait()
        result = self.result[task_id]
        del self.result[task_id]
        del self.task_event[task_id]
        return result

    def recv_results(self):
        while True:
            try:
                try:
                    with self.response_socket_lock:
                        frames = self.control_client.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    time.sleep(0.001)
                    continue
                data = frames[-1]
                result = msgpack.unpackb(data)
                task_id = result["task_id"]
                self.result[task_id] = result["result"]
                self.task_event[task_id].set()
            except Exception as e:
                print(f"zmq client occurred error {e} type: {type(e)} frames: {frames}")
