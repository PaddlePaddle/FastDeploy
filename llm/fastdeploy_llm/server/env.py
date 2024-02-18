"""
保存服务有关的内部用的环境资源
"""
import os

# 服务资源的HOME目录
fastdeploy_llm_home = os.path.join(os.path.expanduser('~'), '.fastdeploy_llm')

# 用于保存服务进程的 PGID
pgid_file_path = os.path.join(fastdeploy_llm_home, 'fastdeploy_llm.pgid')

