import fastdeploy as fd
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encrypted_model_dir",
        required=False,
        help="Path of model directory.")
    parser.add_argument(
        "--model_file", required=True, help="Path of model file directory.")
    parser.add_argument(
        "--params_file",
        required=True,
        help="Path of parameters file directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    model_buffer = open(args.model_file, 'rb')
    params_buffer = open(args.params_file, 'rb')
    encrypted_model, key = fd.encryption.encrypt(model_buffer.read())
    # use the same key to encrypt parameter file
    encrypted_params, key = fd.encryption.encrypt(params_buffer.read(), key)
    encrypted_model_dir = "encrypt_model_dir"
    if args.encrypted_model_dir:
        encrypted_model_dir = args.encrypted_model_dir
    model_buffer.close()
    params_buffer.close()
    os.mkdir(encrypted_model_dir)
    with open(os.path.join(encrypted_model_dir, "__model__.encrypted"),
              "w") as f:
        f.write(encrypted_model)

    with open(os.path.join(encrypted_model_dir, "__params__.encrypted"),
              "w") as f:
        f.write(encrypted_params)

    with open(os.path.join(encrypted_model_dir, "encryption_key.txt"),
              "w") as f:
        f.write(key)
    print("encryption key: ", key)
    print("encryption success")
