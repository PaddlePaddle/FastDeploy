
export http_proxy=agent.baidu.com:8188
export https_proxy=agent.baidu.com:8188
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -r requirements.txt

# 卸载旧版本

python -m pip uninstall paddlepaddle-xpu -y
python -m pip uninstall fastdeploy-xpu -y

# 安装PaddlePaddle
export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,pypi.tuna.tsinghua.edu.cn,paddle-ci.gz.bcebos.com
python -m pip install --pre paddlepaddle-xpu -i https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/

# ============ 编译项目 ============


bash custom_ops/xpu_ops/download_dependencies.sh develop
export CLANG_PATH=$(pwd)/custom_ops/xpu_ops/third_party/xtdk
export XVLLM_PATH=$(pwd)/custom_ops/xpu_ops/third_party/xvllm
bash build.sh || exit 1

