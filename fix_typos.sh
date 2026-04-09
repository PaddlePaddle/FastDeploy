#!/bin/bash

# ==========================
# 第二批 typo 自动修复
# ==========================

# orginal → original
sed -i 's/orginal/original/g' ./tests/ci_use/iluvatar_UT/test.jsonl

# filname → filename
sed -i 's/filname/filename/g' ./fastdeploy/input/ernie4_5_vl_processor/utils/io_utils.py

# OUPUT → OUTPUT
sed -i 's/OUPUT/OUTPUT/g' ./custom_ops/xpu_ops/src/plugin/src/kernel/kunlun3cpp/quant2d_per_channel.xpu

# caculate → calculate
sed -i 's/caculate/calculate/g' ./fastdeploy/eplb/experts_manager.py
sed -i 's/caculate/calculate/g' ./tests/eplb/test_experts_manager.py
sed -i 's/caculate/calculate/g' ./fastdeploy/worker/hpu_worker.py

# lengthes → lengths
sed -i 's/lengthes/lengths/g' ./tests/ci_use/iluvatar_UT/test.jsonl

# Triger → Trigger
sed -i 's/Triger/Trigger/g' ./fastdeploy/worker/hpu_worker.py

# hiddden → hidden
sed -i 's/hiddden/hidden/g' ./fastdeploy/worker/hpu_model_runner.py
sed -i 's/hiddden/hidden/g' ./fastdeploy/model_executor/xpu_pre_and_post_process.py

echo "✅ 第二批所有 typo 修复完成！"