#!/bin/bash

# using v0 version, the request must be sent to the decode instance
# using v1 version, the request can be sent to the prefill or decode instance
# using v2 version, the request must be sent to the router

port=${1:-9000}
echo "port: ${port}"

unset http_proxy && unset https_proxy

curl -X POST "http://0.0.0.0:${port}/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "Introduce shenzhen"}
  ],
  "max_tokens": 20,
  "stream": true
}'
