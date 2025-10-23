from huggingface_hub import snapshot_download

repo_id = "baidu/ERNIE-4.5-0.3B-Paddle"
local_dir = "/root/wenlei07/model/ERNIE-4.5-0.3B-Paddle"

# 下载整个仓库的内容到指定目录
local_path = snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    # 启用断点续传和更好的下载管理
    local_dir_use_symlinks=False,
)

print(f"模型文件已下载到: {local_path}")
