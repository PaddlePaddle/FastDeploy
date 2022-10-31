import os
import yaml
import argparse
from rknn.api import RKNN


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", default=True, help="rknntoolkit verbose")
    parser.add_argument("--config_path")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = get_config()
    with open(config.config_path) as file:
        file_data = file.read()
        yaml_config = yaml.safe_load(file_data)
    print(yaml_config)
    model = RKNN(config.verbose)

    # Config
    mean_values = [[255 * mean for mean in yaml_config["normalize"]["mean"]]]
    std_values = [[255 * std for std in yaml_config["normalize"]["std"]]]
    model.config(mean_values=mean_values,
                 std_values=std_values,
                 target_platform=yaml_config["target_platform"])

    # Load ONNX model
    print(type(yaml_config["outputs"]))
    print("yaml_config[\"outputs\"] = ", yaml_config["outputs"])
    if yaml_config["outputs"] == "None":
        ret = model.load_onnx(model=yaml_config["model_path"])
    else:
        ret = model.load_onnx(model=yaml_config["model_path"],
                              outputs=yaml_config["outputs"])
    assert ret == 0, "Load model failed!"

    # Build model
    ret = model.build(do_quantization=None)
    assert ret == 0, "Build model failed!"

    # Init Runtime
    ret = model.init_runtime()
    assert ret == 0, "Init runtime environment failed!"

    # Export
    if not os.path.exists(yaml_config["output_folder"]):
        os.mkdir(yaml_config["output_folder"])

    model_base_name = os.path.basename(yaml_config["model_path"]).split(".")[0]
    model_device_name = yaml_config["target_platform"].lower()
    model_save_name = model_base_name + "_" + model_device_name + ".rknn"
    ret = model.export_rknn(
        os.path.join(yaml_config["output_folder"], model_save_name))
    assert ret == 0, "Export rknn model failed!"
    print("Export OK!")
