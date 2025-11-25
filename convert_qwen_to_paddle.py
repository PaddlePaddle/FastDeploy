
import os
import json
import paddle
from paddlenlp.utils.converter import Converter

# 源 HuggingFace 模型目录（包含 pytorch_model.bin 和 config.json）
SRC_DIR = "/home/data/dyf/hf_cache/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796/"
DST_DIR = "./paddle_qwen_7b"
os.makedirs(DST_DIR, exist_ok=True)

# 初始化 Converter 时传入 input_dir
converter = Converter(input_dir=SRC_DIR)

# 执行转换（返回 state_dict）
state_dict = converter.convert(input_dir=SRC_DIR)

# 保存为 Paddle 权重
paddle.save(state_dict, os.path.join(DST_DIR, "model_state.pdparams"))

# 同步配置文件
cfg_src = os.path.join(SRC_DIR, "config.json")
cfg_dst = os.path.join(DST_DIR, "model_config.json")
if os.path.exists(cfg_src):
    with open(cfg_src, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(cfg_dst, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

print(f"✅ 转换完成，输出在 {DST_DIR}")

