from pathlib import Path
from safetensors import safe_open
import json
import re

def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

def layers_are_grouped(keys):

    seen = set()
    current_layer = None

    for k in keys:
        m = re.search(r"layers\.(\d+)", k)
        if not m:
            continue

        layer = int(m.group(1))

        if layer != current_layer:
            if layer in seen:
                return False
            seen.add(layer)
            current_layer = layer

    return True

def values_are_naturally_ordered(values):
    """Check if values are sorted in natural order."""
    return list(values) == sorted(values, key=natural_key)


def get_all_weights_file(model_path: str):
    """
    get_all_safetensors
    """
    model_path = Path(model_path)
    use_safetensors = True
    files_list = [str(file) for file in model_path.glob("*.pdparams") if file.name != "scheduler.pdparams"]
    if len(files_list) > 0:
        ordered_weight_map = {}
        use_safetensors = False
        # dont care about the order of the files
        return files_list, {}, use_safetensors, False
    else:
        safe_model_path = model_path / "model.safetensors"
        if safe_model_path.exists():
            with safe_open(safe_model_path, framework="np", device="cpu") as f:
                key_name_list = sorted(f.keys(), key=natural_key)
            ordered_weight_map = {key: "model.safetensors" for key in key_name_list}
            is_layers_are_grouped = True
            files_list = [str(safe_model_path)]
            return files_list, ordered_weight_map, use_safetensors, is_layers_are_grouped
        else:
            index_file = model_path / "model.safetensors.index.json"
            with index_file.open("r") as f:
                weight_map = json.load(f)["weight_map"]
            keys = list(weight_map.keys())
            values = list(weight_map.values())
            is_keys_orders = layers_are_grouped(keys)
            is_values_naturally_ordered = values_are_naturally_ordered(values)
            is_layers_are_grouped = is_keys_orders and is_values_naturally_ordered
            ordered_weight_map = {
                key: str(model_path / weight_map[key]) for key in sorted(weight_map.keys(), key=natural_key)
            }
            weight_files_in_index = {str(model_path / weight_map[name]) for name in weight_map}
            files_list = sorted(weight_files_in_index)
            return files_list, ordered_weight_map, use_safetensors, is_layers_are_grouped


files_list, ordered_weight_map, use_safetensors, is_layers_are_grouped = get_all_weights_file("/root/paddlejob/share-storage/gpfs/system-public/dangweichong/models/eb5_v4p1_v4p2_merge_0413v1_step120")

print("is_layers_are_grouped: ", is_layers_are_grouped)