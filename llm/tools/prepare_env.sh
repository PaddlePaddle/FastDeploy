git clone https://github.com/PaddlePaddle/PaddleNLP
git clone https://github.com/PaddlePaddle/FastDeploy
cd FastDeploy && git checkout llm && cd llm
python3 setup.py bdist_wheel
python3 -m pip install dist/*
wget https://bj.bcebos.com/fastdeploy/llm/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl
python3 -m pip install paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl
rm -rf paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl
python3 -m pip install regex
cd /PaddleNLP/ && python3 setup.py bdist_wheel
python3 -m pip install dist/*
cd /PaddleNLP/csrc &&  python3 setup_cuda.py install --user
rm -rf /FastDeploy
rm -rf /PaddleNLP
