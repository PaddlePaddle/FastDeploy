export PYTHONPATH=/root/paddlejob/workspace/env_run/output/lizexu/FastDeploy-1

# 设置环境变量（指向已启动的服务地址）
export URL=http://localhost:8180/v1/chat/completions
export TEMPLATE=TOKEN_LOGPROB

python -m pytest -sv tests/ce/server/test_base_chat.py
