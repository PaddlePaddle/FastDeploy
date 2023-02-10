import fastdeploy as fd
import os

def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Path of model directory.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    model_file = os.path.join(args.model, "inference.pdmodel")
    params_file = os.path.join(args.model, "inference.pdiparams")
    config_file = os.path.join(args.model, "inference_cls.yaml")
    model_buffer = open(model_file, 'rb')
    params_buffer = open(params_file, 'rb')
    encrypted_model, key = fd.encryption.encrypt(model_buffer.read())
    encrypted_params, key= fd.encryption.encrypt(params_buffer.read(), key)
    encrypted_model_dir = args.model + "_encrypt"
    model_buffer.close()
    params_buffer.close()
    os.mkdir(encrypted_model_dir)
    with open(os.path.join(encrypted_model_dir, "__model__.encrypted"), "w") as f:
        f.write(encrypted_model)

    with open(os.path.join(encrypted_model_dir, "__params__.encrypted"), "w") as f:
        f.write(encrypted_params)

    with open(os.path.join(encrypted_model_dir, "encryption_key.txt"), "w") as f:
        f.write(key)
    print("encryption success")