#!/bin/bash

# ==========================
# 第三批 typo 自动修复
# ==========================

# respose → response
sed -i 's/respose/response/g' ./fastdeploy/entrypoints/openai/v1/serving_chat.py
sed -i 's/respose/response/g' ./fastdeploy/entrypoints/openai/v1/serving_completion.py

# decalare → declare
sed -i 's/decalare/declare/g' ./fastdeploy/model_executor/ops/triton_ops/triton_utils_v2.py

# lanuch → launch
sed -i 's/lanuch/launch/g' ./fastdeploy/model_executor/ops/triton_ops/triton_utils.py
sed -i 's/lanuch/launch/g' ./fastdeploy/model_executor/ops/triton_ops/triton_utils_v2.py


echo "✅ 第三批所有 typo 修复完成！"