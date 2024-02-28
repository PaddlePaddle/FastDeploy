"""
Command line entrypoint for scheduler
"""
import argparse
import multiprocessing
import os
import signal
import json

from .env import fastdeploy_llm_home
from .env import pgid_file_path


if not os.path.exists(fastdeploy_llm_home):
    os.mkdir(fastdeploy_llm_home)


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser("Fastdeploy llm launcher")
    parser.add_argument("--http-port", type=int, default=8100)
    parser.add_argument('cmd', nargs='?', help='command for fastdeploy_llm')
    args = parser.parse_args()
    if args.cmd == 'stop':
        if os.path.exists(pgid_file_path):
            with open(pgid_file_path, 'r') as f:
                pgid = f.read().strip()  
                # 发送 SIGINT 信号以停止服务  
                os.killpg(int(pgid), signal.SIGINT)  
        return
    if os.getenv("MODEL_DIR", None) is None:
        raise ValueError("Environment variable MODEL_DIR must be set")
    from .app import run
    # 服务启动时重置服务需要的资源文件
    if os.path.exists(pgid_file_path):
        os.remove(pgid_file_path)
    with open(pgid_file_path, 'w') as f:
        f.write(str(os.getpgid(os.getpid()))) # 获取进程组pgid
    run(args)

if __name__ == '__main__':
    os.setsid()
    main()