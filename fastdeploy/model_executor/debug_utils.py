# fastdeploy/model_executor/debug_utils.py

import paddle
import pprint
from paddleformers.utils.log import logger
import os

# 全局开关，通过环境变量控制
ENABLE_FULL_DEBUG = os.environ.get("FD_FULL_DEBUG", "0") == "1"

def full_debug_print(obj, name, header=None):
    if not ENABLE_FULL_DEBUG:
        return

    if header:
        logger.warning("\n" + "#"*30 + f" {header} " + "#"*30)

    title = f"DEBUG DUMP: {name}"
    header_line = f"\n{'='*25} {title} {'='*25}"
    footer_line = "=" * (52 + len(title)) + "\n"
    
    content = ""
    try:
        if obj is None:
            content = "  - Is None"
        elif isinstance(obj, paddle.Tensor):
            with paddle.no_grad():
                shape = list(obj.shape)
                dtype = str(obj.dtype)
                numel = obj.numel()
                place = str(obj.place)
                stats = {
                    'shape': shape,
                    'dtype': dtype,
                    'numel': numel,
                    'place': place,
                }
                if numel > 0 and numel < 20: # 如果元素不多，直接打印全部
                    stats['values'] = obj.cpu().numpy().tolist()
                elif numel > 0:
                    stats['first_5_values'] = obj.flatten().cpu().numpy()[:5].tolist()
                    tensor_float = obj.astype('float32').cpu()
                    stats["max"] = f"{tensor_float.max().item():.6f}"
                    stats["min"] = f"{tensor_float.min().item():.6f}"
                    stats["mean"] = f"{tensor_float.mean().item():.6f}"

                content = pprint.pformat(stats, indent=2)
        elif isinstance(obj, (int, float, bool, str, list, dict, tuple)):
            content = pprint.pformat(obj, indent=2)
        else:
            content = f"  - Type: {type(obj)}, cannot print details."
    except Exception as e:
        content = f"  - FAILED TO PRINT: {e}"

    logger.warning(header_line + "\n" + content + "\n" + footer_line)