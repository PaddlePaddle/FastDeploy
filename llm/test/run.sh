#!/bin/bash

current_directory=$PWD
pip uninstall -y paddlepaddle-gpu
pip uninstall -y paddlenlp
unset http_proxy
unset https_proxy
wget https://bj.bcebos.com/fastdeploy/llm/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl
pip install paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl
export https_proxy=http://172.19.56.199:3128
export http_proxy=http://172.19.56.199:3128
git clone https://github.com/PaddlePaddle/PaddleNLP.git
git clone -b llm https://github.com/PaddlePaddle/FastDeploy.git
pip install wget
unset http_proxy
unset https_proxy
cd PaddleNLP
python3 setup.py bdist_wheel
cd dist
pip install $(ls)
cd ..
cd csrc
python3 setup_cuda.py install --user
cd $current_directory
python3 -u ci.py
result=$?
if [ $result -eq 0 ];then
  echo "通过测试"
else
  echo "测试失败"
fi
echo "具体结果如下："
cat results.txt
exit $result
