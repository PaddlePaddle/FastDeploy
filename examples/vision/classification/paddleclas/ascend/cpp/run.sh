#!/bin/bash

# 模型名字
MODEL_NAME=ResNet50_vd_infer
# 预测的图片名字
DATA_NAME=ILSVRC2012_val_00000010.jpeg
# 本demo的可执行文件
DEMO_NAME=infer_demo

export GLOG_v=5
# 设置本demo的环境变量
# 正确设置fastdeploy-cann的安装路径
FASTDEPLOY_INSTALL_DIR="../../../../../../build/fastdeploy-cann/"
# 设置fastdeploy,opencv和paddlelite相关的环境变量
export LD_LIBRARY_PATH=$FASTDEPLOY_INSTALL_DIR/lib/:$FASTDEPLOY_INSTALL_DIR/third_libs/install/opencv/lib/:$FASTDEPLOY_INSTALL_DIR/third_libs/install/paddlelite/lib/:$LD_LIBRARY_PATH

# 设置昇腾相关环境变量
HUAWEI_ASCEND_TOOLKIT_HOME="/usr/local/Ascend/ascend-toolkit/latest"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/stub:$HUAWEI_ASCEND_TOOLKIT_HOME/acllib/lib64:$HUAWEI_ASCEND_TOOLKIT_HOME/atc/lib64:$HUAWEI_ASCEND_TOOLKIT_HOME/opp/op_proto/built-in
export PYTHONPATH=$PYTHONPATH:$HUAWEI_ASCEND_TOOLKIT_HOME/fwkacllib/python/site-packages:$HUAWEI_ASCEND_TOOLKIT_HOME/acllib/python/site-packages:$HUAWEI_ASCEND_TOOLKIT_HOME/toolkit/python/site-packages:$HUAWEI_ASCEND_TOOLKIT_HOME/atc/python/site-packages:$HUAWEI_ASCEND_TOOLKIT_HOME/pyACL/python/site-packages/acl
export PATH=$PATH:$HUAWEI_ASCEND_TOOLKIT_HOME/atc/ccec_compiler/bin:${HUAWEI_ASCEND_TOOLKIT_HOME}/acllib/bin:$HUAWEI_ASCEND_TOOLKIT_HOME/atc/bin
export ASCEND_AICPU_PATH=$HUAWEI_ASCEND_TOOLKIT_HOME
export ASCEND_OPP_PATH=$HUAWEI_ASCEND_TOOLKIT_HOME/opp
export TOOLCHAIN_HOME=$HUAWEI_ASCEND_TOOLKIT_HOME/toolkit
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3

chmod +x ./$BUILD_DIR

# 运行本demo.
./build/$DEMO_NAME ./models/$MODEL_NAME ./images/$DATA_NAME
