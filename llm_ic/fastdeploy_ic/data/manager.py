
import json
import math
import asyncio

import aioredis

import fastdeploy_ic.proto.ic_pb2 as ic_pb2
from fastdeploy_ic.utils import get_logger

logger = get_logger("data_manager", "ic_data_manager.log")

__retry_times = 5  # redis client may have unexpected errors, we retry it with respect to some errors
def retry_wrapper(f):
    async def wrapper(*args, **kwargs):
        for i in range(__retry_times):
            try:
                return await f(*args, **kwargs)
            except asyncio.CancelledError:
                logger.info("{} occured asyncio.CancelledError, retry times: {}".format(f.__name__, i+1))
                continue
    return wrapper



class DataManager:
    def __init__(self, redis_conf) -> None:
        self.client = aioredis.Redis(**redis_conf)
        self.internal_check_key_prefix = '__keymap_'
    
    @retry_wrapper
    async def check_req_id_exist(self, model_id, req_id):
        key = '{}{}'.format(self.internal_check_key_prefix, model_id)
        logger.info("check_req_id_exist:  key: {} value: {}".format(key, req_id))
        is_exist = await self.client.sismember(key, req_id)
        return is_exist

    @retry_wrapper
    async def add_req_id_to_map(self, model_id, req_id):
        key = '{}{}'.format(self.internal_check_key_prefix, model_id)
        logger.info("add_req_id_to_map:  key: {} value: {}".format(key, req_id))
        await self.client.sadd(key, req_id)

    @retry_wrapper
    async def remove_req_id_from_map(self, model_id, req_id):
        key = '{}{}'.format(self.internal_check_key_prefix, model_id)
        logger.info("remove_req_id_from_map:  key: {} value: {}".format(key, req_id))
        await self.client.srem(key, req_id)

    @retry_wrapper
    async def enque_request(self, model_id, req, to_end=True):
        serialized_req = req.SerializeToString()
        # key = model_id
        logger.info("enque_request:  key: {} value: {}".format(model_id, req))
        if to_end:
            await self.client.rpush(model_id, serialized_req)
        else:
            await self.client.lpush(model_id, serialized_req)

    @retry_wrapper  
    async def deque_request(self, model_id):
        data = await self.client.lpop(model_id)
        if data is not None:
            data = ic_pb2.ModelInferRequest.FromString(data)
        logger.info("deque_request:  key: {} value: {}".format(model_id, data))
        return data

    @retry_wrapper  
    async def remove_request(self, model_id, req):
        serialized_req = req.SerializeToString()
        logger.info("remove_request:  key: {} value: {}".format(model_id, req))
        await self.client.lrem(model_id, 1, serialized_req)

    @retry_wrapper
    async def enque_response(self, model_id, req_id, res, to_end=True):
        serialized_res = res.SerializeToString()
        key = '{}/{}'.format(model_id, req_id)
        logger.info("enque_response:  key: {} value: {}".format(key, res))
        if to_end:
            await self.client.rpush(key, serialized_res)
        else:
            await self.client.lpush(key, serialized_res)

    @retry_wrapper
    async def deque_response(self, model_id, req_id):
        key = '{}/{}'.format(model_id, req_id)
        data = await self.client.lpop(key)
        if data is not None:
            data = ic_pb2.ModelInferResponse.FromString(data)
        logger.info("deque_response:  key: {} value: {}".format(key, data))
        return data
    
    @retry_wrapper
    async def clear_response(self, model_id, req_id):
        key = '{}/{}'.format(model_id, req_id)
        logger.info("clear_response:  key: {}".format(key))
        await self.client.delete(key)

    async def get_requests_by_number(self, model_id, max_request_num):
        # return requests by ByRequest strategy
        requests = []
        for i in range(max_request_num):
            request = await self.deque_request(model_id)
            if request is not None:
                requests.append(request)
            else:
                break
        logger.info("get_requests_by_number:  model_id: {} length: {}".format(model_id, len(requests)))
        return requests

    async def get_requests_by_block(self, model_id, max_request_num, block_num, block_size, dec_token_num):
        # return requests by ByToken strategy
        requests = []
        left_block_num = block_num
        for i in range(max_request_num):
            request = await self.deque_request(model_id)
            if request is not None:
                text_words_num = json.loads(request.input)['text_words_num']
                need_block_num = math.ceil((text_words_num + dec_token_num)/block_size)
                if need_block_num < left_block_num:
                    requests.append(request)
                    left_block_num -= need_block_num
                else:
                    await self.enque_request(model_id, request, to_end=False)
                    break
        logger.info("get_requests_by_block:  model_id: {} length: {}".format(model_id, len(requests)))
        return requests