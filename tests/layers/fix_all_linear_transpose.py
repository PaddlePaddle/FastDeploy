import paddle
import paddle.nn as nn

def fix_all_linear_layers(model):
    replacements = []
    for name, layer in model.named_sublayers():
        if isinstance(layer, nn.Linear):
            replacements.append((name, layer))

    print(f"[Info] Found {len(replacements)} Linear layers. ")

    # import pdb; pdb.set_trace()
    for name, layer in replacements:
        # print(f"[Fix] Rebuilding {name}: shape {list(layer.weight.shape)} -> {list(layer.weight.T.shape)}")
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)

        in_features = layer.weight.shape[0]
        out_features = layer.weight.shape[1]

        # 注意: 原层的 weight 是 [in, out]，但正确应是 [out, in]
        new_layer = nn.Linear(
            in_features=out_features,  # ✅ 交换
            out_features=in_features,  # ✅ 交换
            bias_attr=layer.bias is not None
        )

        # 转 dtype & 复制参数
        dtype = str(layer.weight.dtype)
        dtype = dtype.replace("paddle.", "")
        new_layer = new_layer.astype(dtype)

        w_new = layer.weight.T.astype(dtype).clone()
        new_layer.weight = paddle.create_parameter(shape=w_new.shape, dtype=dtype,  default_initializer = nn.initializer.Constant(0.0) )
        new_layer.weight.set_value(w_new)

        if layer.bias is not None:
            b_new = layer.bias.astype(dtype).clone()
            new_layer.bias = paddle.create_parameter(shape=b_new.shape, dtype=dtype,  default_initializer = nn.initializer.Constant(0.0) )
            new_layer.bias.set_value(b_new)

        setattr(parent, parts[-1], new_layer)

    print("[Done] All Linear layers have been rebuilt and replaced.")
    return model
