# 代码提交说明

FastDeploy使用clang-format, cpplint检查和格式化代码，提交代码前，需安装pre-commit
```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
git checkout develop

pip install pre-commit
pip install yapf
pip install cpplint
pre-commit install
```
commit代码时，若提示无法找到clang-format，请自行安装clang-format
