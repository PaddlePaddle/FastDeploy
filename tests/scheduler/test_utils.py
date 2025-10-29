"""
Test utilities for scheduler tests
"""
import time
import uuid
from unittest.mock import Mock, MagicMock
from fastdeploy.engine.request import Request, RequestOutput, CompletionOutput, RequestMetrics


def create_mock_request(request_id=None, prompt_token_ids_len=100, arrival_time=None):
    """Create a mock Request object for testing"""
    if request_id is None:
        request_id = str(uuid.uuid4())
    if arrival_time is None:
        arrival_time = time.time()
    
    request = Mock(spec=Request)
    request.request_id = request_id
    request.prompt_token_ids_len = prompt_token_ids_len
    request.arrival_time = arrival_time
    request.disaggregate_info = None
    request.to_dict.return_value = {
        "request_id": request_id,
        "prompt_token_ids_len": prompt_token_ids_len,
        "arrival_time": arrival_time,
        "disaggregate_info": None
    }
    return request


def create_mock_request_output(request_id, finished=False, error_code=200, send_idx=0):
    """Create a mock RequestOutput object for testing"""
    output = Mock(spec=RequestOutput)
    output.request_id = request_id
    output.finished = finished
    output.error_code = error_code
    output.outputs = Mock(spec=CompletionOutput)
    output.outputs.send_idx = send_idx
    output.to_dict.return_value = {
        "request_id": request_id,
        "finished": finished,
        "error_code": error_code,
        "outputs": {"send_idx": send_idx}
    }
    return output


def create_mock_redis_client():
    """Create a mock Redis client for testing"""
    redis_client = Mock()
    redis_client.hgetall.return_value = {}
    redis_client.hset.return_value = True
    redis_client.hdel.return_value = True
    redis_client.lpush.return_value = True
    redis_client.rpop.return_value = None
    redis_client.brpop.return_value = None
    redis_client.pipeline.return_value.__enter__.return_value = redis_client
    redis_client.multi.return_value = None
    redis_client.execute.return_value = [True]
    redis_client.expire.return_value = True
    return redis_client


def create_mock_config(**kwargs):
    """Create a mock SplitWiseSchedulerConfig for testing"""
    config = Mock()
    config.nodeid = kwargs.get('nodeid', str(uuid.uuid4()))
    config.redis_host = kwargs.get('redis_host', '127.0.0.1')
    config.redis_port = kwargs.get('redis_port', 6379)
    config.redis_password = kwargs.get('redis_password', None)
    config.redis_topic = kwargs.get('redis_topic', 'fd')
    config.ttl = kwargs.get('ttl', 900)
    config.release_load_expire_period = kwargs.get('release_load_expire_period', 600)
    config.sync_period = kwargs.get('sync_period', 5)
    config.expire_period = kwargs.get('expire_period', 3.0)
    config.clear_expired_nodes_period = kwargs.get('clear_expired_nodes_period', 60)
    config.reader_parallel = kwargs.get('reader_parallel', 4)
    config.reader_batch_size = kwargs.get('reader_batch_size', 200)
    config.writer_parallel = kwargs.get('writer_parallel', 4)
    config.writer_batch_size = kwargs.get('writer_batch_size', 200)
    config.max_model_len = kwargs.get('max_model_len', 4096)
    config.enable_chunked_prefill = kwargs.get('enable_chunked_prefill', True)
    config.max_num_partial_prefills = kwargs.get('max_num_partial_prefills', 4)
    config.max_long_partial_prefills = kwargs.get('max_long_partial_prefills', 2)
    config.long_prefill_token_threshold = kwargs.get('long_prefill_token_threshold', 164)
    return config
