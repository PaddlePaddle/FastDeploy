import time

import grpc
import json
import asyncio
from aioredis import RedisError

import fastdeploy_ic.proto.ic_pb2_grpc as ic_pb2_grpc
import fastdeploy_ic.proto.ic_pb2 as ic_pb2
from fastdeploy_ic.data.manager import DataManager
from fastdeploy_ic.config import GlobalConfig
from fastdeploy_ic.utils import get_logger

logger = get_logger("ic_server", "ic_server.log")

global_config = GlobalConfig()
redis_config = {
  'host': global_config.redis_host,
  'port': global_config.redis_port,
  'db': global_config.redis_db,
  'username': global_config.redis_username, 
  'password': global_config.redis_password
}
data_manager = DataManager(redis_config)

class GRPCInferenceServiceServicer(ic_pb2_grpc.GRPCInferenceServiceServicer):
  async def ModelStreamInfer(self, request, context):
    """
    Provided for request sender.
    """
    try:
      model_id = request.model_id
      input_dict = json.loads(request.input)
      if 'req_id' not in input_dict:
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "ModelStreamInfer: there is no req_id in request")
      if 'ic_req_data' not in input_dict:
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "ModelStreamInfer: there is no ic_req_data in request")
      req_id = input_dict['req_id']
      # Check whether req_id is repeated
      # Warning: We only simply check whether there is any same req_id has been in, 
      #   but we can not prevent two requests with the same req_id coming simultaneously.
      #   To achieve this, we should add lock to query and insert query into redis, which will influence performance. 
      #   Currently, we assume different req_ids are confirmed by users.
      if await data_manager.check_req_id_exist(model_id, req_id):
        logger.info("ModelStreamInfer: req_id {}: has existed in other task".format(req_id))
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "ModelStreamInfer: req_id {}: has existed in other task".format(req_id))
      # 1. push request to redis
      await data_manager.add_req_id_to_map(model_id, req_id)
      await data_manager.enque_request(model_id, request)
      # 2. response stream results
      response_start_time = time.time()
      while True:
        if time.time() - response_start_time > global_config.resonpse_timeout:
            if await data_manager.check_req_id_exist(model_id, req_id):  # clear resource about this req
              await data_manager.remove_request(model_id, request)
              await data_manager.clear_response(model_id, req_id)
              await data_manager.remove_req_id_from_map(model_id, req_id)
            logger.info("ModelStreamInfer: req_id {}: Get response from inference server timeout".format(req_id))
            await context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "ModelStreamInfer: req_id {}: Get response from inference server timeout".format(req_id))
        data = await data_manager.deque_response(model_id, req_id)
        if data is None:
          await asyncio.sleep(1)
          continue
        try:
          output_dict = json.loads(data.output)
          if 'ic_timestamp_tag' in output_dict:
            if time.time() - output_dict['ic_timestamp_tag'] > global_config.resonpse_timeout: # the response is invalid because of timeout, even maybe from previous request with same req_id
              continue
            del output_dict['ic_timestamp_tag']
            data.output = json.dumps(output_dict)
          logger.info("ModelStreamInfer: req_id {}: response data: {}".format(req_id, output_dict))
          yield data
          # two cases denote the request is done
          # 1. something error returned by server, but not normal result
          # 2. is_end is 1
          if ('is_end' not in output_dict) or (output_dict['is_end'] == 1): 
            # clear resource about this req, only req_id in map should be removed
            await data_manager.remove_req_id_from_map(model_id, req_id) 
            return
          
        except Exception as e:
          if await data_manager.check_req_id_exist(model_id, req_id):  # clear resource about this req
              await data_manager.clear_response(model_id, req_id)
              await data_manager.remove_req_id_from_map(model_id, req_id)
          logger.info("ModelStreamInfer: req_id {}: Failed to read response data from inference server, exception {}".format(req_id, e))
          await context.abort(grpc.StatusCode.INTERNAL, "ModelStreamInfer: req_id {}: Failed to read response data from inference server".format(req_id))
    except RedisError as e:
      # if redis operation failed, should arrive here    
      # Log the error message, and signal users internal error (we can not expose origin redis error to users)
      logger.info("ModelStreamInfer: exception: {}".format(e))
      await context.abort(grpc.StatusCode.INTERNAL, "Internal error happened")

  async def ModelFetchRequest(self, request, context):
    """
    Provide for inference service.
    """
    # provide two types for providing tasks
    # 1. ByRequest
    # 2. ByToken
    try:
      model_ids = request.model_id
      strategy = request.strategy
      requests = []
      for model_id in model_ids:
        if strategy == ic_pb2.FetchStrategy.ByRequest:
          requests.extend(await data_manager.get_requests_by_number(model_id, request.max_request_num))
          
        else:
          by_token_params = request.by_token_params
          requests.extend(await data_manager.get_requests_by_block(model_id, request.max_request_num,
                              by_token_params.block_num, by_token_params.block_size, by_token_params.dec_token_num))
      
      fetch_request_result = ic_pb2.ModelFetchRequestResult()
      fetch_request_result.requests.extend(requests)
      logger.info("ModelFetchRequest: return requests: {}".format(requests))
    except RedisError as e:
      # if operation failed, should arrive here    
      # Log the error message, and signal users internal error
      logger.info("ModelFetchRequest: exception: {}".format(e))
      await context.abort(grpc.StatusCode.INTERNAL, "Internal error happened")
    return fetch_request_result


  async def ModelSendResponse(self, response_iterator, context):
    """
    Provide for inference service.
    """
    # Get response from inference server
    try:
      response_start_time = time.time()
      async for response in response_iterator:
        try:
          res = json.loads(response.output)
          model_id = res['ic_req_data']['model_id']
          req_id = res['req_id']
          # add timestamp for response
          res['ic_timestamp_tag'] = time.time()  # we add this to prevent that client recieves 
                                    # response for previous request due to: 
                                    # 1. use the same req_id by mistake 
                                    # 2. the client corresponding to previous request did not recieve all responses for some reason  
          response.output = json.dumps(res)
        except:
          logger.info("ModelSendResponse: req_id {}: Failed to read response data from inference server".format(req_id))
          await context.abort(grpc.StatusCode.INTERNAL, "ModelSendResponse: req_id {}: Failed to read response data from inference server".format(req_id))
        await data_manager.enque_response(model_id, req_id, response)
        logger.info("ModelSendResponse: req_id {}: response data: {}".format(req_id, res))
        if ('is_end' not in res) or (res['is_end'] == 1):
            return ic_pb2.ModelSendResponseResult()
        if time.time() - response_start_time > global_config.resonpse_timeout:
          await data_manager.clear_response(model_id, req_id)
          logger.info("ModelSendResponse: req_id {}: Get response from inference server timeout".format(req_id))
          await context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "ModelSendResponse: req_id {}: Get response from inference server timeout".format(req_id))
    except RedisError as e:
      # if operation failed, should arrive here    
      # Log the error message, and signal users internal error
      logger.info("ModelSendResponse: exception: {}".format(e))
      await context.abort(grpc.StatusCode.INTERNAL, "Internal error happened")

  async def ModelSendResponseList(self, response_list_iterator, context):
    """
    Provide for inference service.
    """
    # Get response from inference server
    try:
      response_start_time = time.time()
      async for response_list in response_list_iterator:
        for response in response_list.response_list:
          try:
            res = json.loads(response.output)
            model_id = res['ic_req_data']['model_id']
            req_id = res['req_id']
            # add timestamp for response
            res['ic_timestamp_tag'] = time.time()  # we add this to prevent that client recieves 
                                    # response for previous request due to: 
                                    # 1. use the same req_id by mistake 
                                    # 2. the client corresponding to previous request did not recieve all responses for some reason  
            response.output = json.dumps(res)
          except:
            logger.info("ModelSendResponseList: req_id {}: Failed to read response data from inference server".format(req_id))
            await context.abort(grpc.StatusCode.INTERNAL, "ModelSendResponseList: req_id {}: Failed to read response data from inference server".format(req_id))
          await data_manager.enque_response(model_id, req_id, response)
          logger.info("ModelSendResponseList: req_id {}: response data: {}".format(req_id, res))
          if ('is_end' not in res) or (res['is_end'] == 1):
              break
          if time.time() - response_start_time > global_config.resonpse_timeout:
            await data_manager.clear_response(model_id, req_id)
            logger.info("ModelSendResponseList: req_id {}: Get response from inference server timeout".format(req_id))
            await context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "ModelSendResponseList: req_id {}: Get response from inference server timeout".format(req_id))
    except RedisError as e:
      # if operation failed, should arrive here    
      # Log the error message, and signal users internal error
      logger.info("ModelSendResponseList: exception: {}".format(e))
      await context.abort(grpc.StatusCode.INTERNAL, "Internal error happened")
    return ic_pb2.ModelSendResponseResult()


