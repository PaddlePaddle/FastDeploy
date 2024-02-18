"""
Http FastAPI app
"""
import asyncio
import subprocess
import os

import uvicorn
from fastapi import FastAPI, APIRouter

from .api import watch_result, model_executor

check_live_path = '/ready'
inference_path = '/v1/chat/completions' 

def create_app():
    """
    Create a FastAPI app.
    """
    app = FastAPI()
    router = APIRouter()
    url_mappings = [  
        (check_live_path, model_executor.check_live, ["GET"]),  
        (inference_path, model_executor.inference, ["POST"]),  
    ]  
    for url, view_func, supported_methods in url_mappings:  
        router.add_api_route(url, endpoint=view_func, methods=supported_methods)
    app.include_router(router)
    return app

app = create_app()

# FastAPI 的启动事件  
@app.on_event("startup")  
async def startup_event():
    """
    监控结果是否产生
    """
    model_executor.prepare_model()
    watch_result_task = asyncio.create_task(watch_result())

def run(args):
    """
    start http server
    """
    uvicorn.run("fastdeploy_llm.server.app:app", host="0.0.0.0", port=int(args.http_port), log_level="info")

    
    