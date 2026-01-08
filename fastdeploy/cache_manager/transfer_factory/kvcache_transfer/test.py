"""
CacheMessager test:
* prefill gets block ids from decode
* prefill sets cache values
* prefill sends cache to decode
* decode receives cache and validates cache values
"""

import argparse
import math
import random
import time

import paddle
import rdma_comm
import zmq

if paddle.is_compiled_with_xpu():
    from custom_setup_ops import get_peer_mem_addr


class CacheMessager(object):
    def __init__(self, splitwise_role, num_layers, max_block_num, port=None, device="gpu"):
        assert splitwise_role in ["prefill", "decode"], "splitwise_role must be prefill or decode"
        if splitwise_role == "decode":
            assert port, "port must be specified for decode server"
        self.splitwise_role = splitwise_role
        self.num_layers = num_layers
        self.max_block_num = max_block_num
        self.gpu_cache_kvs = {}
        paddle.device.set_device(device)
        print(f"splitwise role: {splitwise_role}, port: {port}, device: {device}")

        if paddle.is_compiled_with_xpu():
            cache_type = "float16"
        else:
            cache_type = "bfloat16"  # bfloat16 or uint8
        kv_num_head = 2
        block_size_seq_len = 64
        hidden_size = 1024
        num_attention_heads = 8

        cache_k_ptr_list = []
        cache_v_ptr_list = []
        cache_k_scale_ptr_list = []
        cache_v_scale_ptr_list = []
        for layer_idx in range(num_layers):
            key_cache = paddle.full(
                shape=[
                    max_block_num,
                    kv_num_head,
                    block_size_seq_len,
                    hidden_size // num_attention_heads,
                ],
                fill_value=-1,
                dtype=cache_type,
            )
            self.gpu_cache_kvs[f"key_caches_{layer_idx}"] = key_cache
            if paddle.is_compiled_with_xpu():
                key_cache_ptr = get_peer_mem_addr(key_cache.data_ptr())
                cache_k_ptr_list.append(key_cache_ptr)
            else:
                cache_k_ptr_list.append(key_cache.data_ptr())
            value_cache = paddle.full(
                shape=[
                    max_block_num,
                    kv_num_head,
                    block_size_seq_len,
                    hidden_size // num_attention_heads,
                ],
                fill_value=-1,
                dtype=cache_type,
            )
            self.gpu_cache_kvs[f"value_caches_{layer_idx}"] = value_cache
            if paddle.is_compiled_with_xpu():
                value_cache_ptr = get_peer_mem_addr(value_cache.data_ptr())
                cache_v_ptr_list.append(value_cache_ptr)
            else:
                cache_v_ptr_list.append(value_cache.data_ptr())

            # Create scale
            key_scale = paddle.full(
                shape=[
                    max_block_num,
                    kv_num_head,
                    block_size_seq_len,
                ],
                fill_value=0.0,
                dtype="float32",
            )
            self.gpu_cache_kvs[f"key_scale_{layer_idx}"] = key_scale
            if paddle.is_compiled_with_xpu():
                key_scale_ptr = get_peer_mem_addr(key_scale.data_ptr())
                cache_k_scale_ptr_list.append(key_scale_ptr)
            else:
                cache_k_scale_ptr_list.append(key_scale.data_ptr())

            value_scale = paddle.full(
                shape=[
                    max_block_num,
                    kv_num_head,
                    block_size_seq_len,
                ],
                fill_value=0.0,
                dtype="float32",
            )
            self.gpu_cache_kvs[f"value_scale_{layer_idx}"] = value_scale
            if paddle.is_compiled_with_xpu():
                value_scale_ptr = get_peer_mem_addr(value_scale.data_ptr())
                cache_v_scale_ptr_list.append(value_scale_ptr)
            else:
                cache_v_scale_ptr_list.append(value_scale.data_ptr())

            if self.splitwise_role == "prefill":
                for block_idx in range(max_block_num):
                    key_cache[block_idx] = block_idx
                    value_cache[block_idx] = block_idx
                    key_scale[block_idx] = block_idx * 0.1  # example scale value
                    value_scale[block_idx] = block_idx * 0.1  # example scale value

        # create messager
        cache_shape = key_cache.shape
        max_block_num = cache_shape[0]
        block_bytes = math.prod(cache_shape[1:]) * (
            2 if key_cache.dtype == paddle.bfloat16 or key_cache.dtype == paddle.float16 else 1
        )

        scale_cache_shape = key_scale.shape
        scale_block_bytes = math.prod(scale_cache_shape[1:]) * 4  # float32 is 4 bytes

        print(
            f"cache_shape: {cache_shape}, max_block_num: {max_block_num}, "
            f"block_bytes: {block_bytes}, scale_block_bytes: {scale_block_bytes}, "
            f"dtype: {key_cache.dtype}"
        )
        self.rdma_comm = rdma_comm.RDMACommunicator(
            splitwise_role,
            0,
            str(port) if self.splitwise_role == "decode" else "0",
            cache_k_ptr_list,
            cache_v_ptr_list,
            max_block_num,
            block_bytes,
            cache_k_scale_ptr_list,
            cache_v_scale_ptr_list,
            scale_block_bytes,
        )

    def is_connected(self, ip, port):
        assert self.splitwise_role == "prefill", "only prefill can call this method"
        return self.rdma_comm.is_connected(ip, str(port))

    def connect(self, ip, port):
        assert self.splitwise_role == "prefill", "only prefill can call this method"
        if self.is_connected(ip, port):
            return True

        self.rdma_comm.connect(ip, str(port))
        return True

    def write_cache(self, ip, port, src_block_ids, dest_block_ids):
        assert self.splitwise_role == "prefill", "only prefill can call this method"
        if not self.is_connected(ip, port):
            raise Exception("Not connected yet")

        print("start write cache to decode")
        for layer_idx in range(self.num_layers):
            self.rdma_comm.write_cache(ip, str(port), src_block_ids, dest_block_ids, layer_idx)
        return True

    def check_cache(self, block_ids, values):
        status = True
        all_details = []
        for layer_idx in range(self.num_layers):
            key_cache = self.gpu_cache_kvs[f"key_caches_{layer_idx}"]
            value_cache = self.gpu_cache_kvs[f"value_caches_{layer_idx}"]
            key_scale = self.gpu_cache_kvs[f"key_scale_{layer_idx}"]
            value_scale = self.gpu_cache_kvs[f"value_scale_{layer_idx}"]
            details = {"layer_idx": layer_idx, "flags": []}
            all_details.append(details)
            # print(f"key_cache of layer_idx: {layer_idx} \n {key_cache[block_ids]}", flush=True)

            for idx in range(len(block_ids)):
                key_flag = bool(paddle.all(key_cache[block_ids[idx]].equal(values[idx])))
                value_flag = bool(paddle.all(value_cache[block_ids[idx]].equal(values[idx])))
                key_scale_flag = bool(paddle.all(key_scale[block_ids[idx]].equal(values[idx] * 0.1)))
                value_scale_flag = bool(paddle.all(value_scale[block_ids[idx]].equal(values[idx] * 0.1)))
                if not (key_flag and value_flag and key_scale_flag and value_scale_flag):
                    status = False
                details["flags"].append((key_flag, value_flag))

        return status, all_details

    def set_cache(self, block_ids, values):
        for layer_idx in range(self.num_layers):
            key_cache = self.gpu_cache_kvs[f"key_caches_{layer_idx}"]
            value_cache = self.gpu_cache_kvs[f"value_caches_{layer_idx}"]
            key_scale = self.gpu_cache_kvs[f"key_scale_{layer_idx}"]
            value_scale = self.gpu_cache_kvs[f"value_scale_{layer_idx}"]
            for idx in range(len(block_ids)):
                key_cache[block_ids[idx]] = values[idx]
                value_cache[block_ids[idx]] = values[idx]
                key_scale[block_ids[idx]] = values[idx] * 0.1
                value_scale[block_ids[idx]] = values[idx] * 0.1
        if paddle.is_compiled_with_cuda():
            paddle.device.cuda.synchronize()
        elif paddle.is_compiled_with_xpu():
            paddle.device.xpu.synchronize()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splitwise_role", type=str, default="prefill", help="prefill or decode")
    parser.add_argument("--decode_ip", type=str, default="0.0.0.0")
    parser.add_argument("--decode_rdma_port", type=int, default=9881)
    parser.add_argument("--decode_zmq_port", type=int, default=9882)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--max_block_num", type=int, default=20)
    parser.add_argument("--send_recv_block_num", type=int, default=5)
    parser.add_argument("--test_num", type=int, default=1)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    splitwise_role = args.splitwise_role
    decode_ip = args.decode_ip
    decode_rdma_port = args.decode_rdma_port
    decode_zmq_port = args.decode_zmq_port
    num_layers = args.num_layers
    max_block_num = args.max_block_num
    send_recv_block_num = args.send_recv_block_num
    test_num = args.test_num
    assert splitwise_role in ["prefill", "decode"], "splitwise_role must be prefill or decode"

    if splitwise_role == "decode":
        context = zmq.Context()
        server_socket = context.socket(zmq.REP)
        server_socket.bind(f"tcp://{decode_ip}:{decode_zmq_port}")
        print(f"zmq server started with port: {decode_zmq_port}")

        cm = CacheMessager("decode", num_layers, max_block_num, port=decode_rdma_port)

        success_num = 0
        fail_num = 0
        for i in range(2 * test_num + 1):
            try:
                print("waiting for request...")
                obj = server_socket.recv_pyobj()
                print(f"recv obj: {obj}")
                if obj["msg_type"] == "get_block_ids":
                    # get block ids
                    all_block_ids = list(range(max_block_num))
                    block_ids = random.sample(all_block_ids, k=send_recv_block_num)
                    server_socket.send_pyobj({"block_ids": block_ids})
                else:
                    # check recv cache
                    block_ids = obj["block_ids"]
                    values = obj["values"]
                    print(f"block_ids: {block_ids}, values: {values}")
                    check_status, check_details = cm.check_cache(block_ids, values)
                    print(f"check_status: {check_status}, check_details: {check_details}")
                    if check_status:
                        success_num += 1
                    else:
                        fail_num += 1
                    if i % 100 == 0:
                        print(f"i: {i}, success_num: {success_num}, fail_num: {fail_num}")
                    server_socket.send_pyobj({"result": "done"})
            except Exception as e:
                print(f"Decode ZMQ server encountered exception: {e}")
        print(f"test_num: {test_num}, success_num: {success_num}")
    else:
        context = zmq.Context()
        client_socket = context.socket(zmq.REQ)
        client_socket.connect(f"tcp://{decode_ip}:{decode_zmq_port}")

        cm = CacheMessager("prefill", num_layers, max_block_num, device="cpu")
        cm.connect(decode_ip, decode_rdma_port)
        while not cm.is_connected(decode_ip, decode_rdma_port):
            time.sleep(1)
            print("wait for connection...")

        for i in range(test_num):
            client_socket.send_pyobj({"msg_type": "get_block_ids"})
            reply = client_socket.recv_pyobj()
            block_ids = reply["block_ids"]
            values = random.sample(list(range(max_block_num)), k=len(block_ids))
            print(f"block_ids: {block_ids}, values: {values}")

            cm.set_cache(block_ids, values)
            cm.write_cache(decode_ip, decode_rdma_port, block_ids, block_ids)
            client_socket.send_pyobj({"msg_type": "check_kv", "block_ids": block_ids, "values": values})
            reply = client_socket.recv_pyobj()

            check_status, check_details = cm.check_cache(block_ids, values)
            print(f"check_status: {check_status}, check_details: {check_details}")
