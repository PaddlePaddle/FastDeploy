# 大模型服务的负载均衡组件

## 环境要求

- python >= 3.7
- 启动好的redis服务，用于作为负载均衡的数据库

## 环境变量
目前所支持的环境变量参考fastdeploy_ic里的config.py

| 环境变量    | 含义 |
| -------- | ------- |
| REDIS_HOST | redis服务的ip |
| REDIS_PORT | redis服务的port |
| REDIS_USERNAME |  redis认证用户         |
| REDIS_PASSWORD |  redis认证密码          |
| RESPONSE_TIMEOUT | 获取推理服务流式token的超时时间 |


## 启动示例

```shell
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
python main.py
```

