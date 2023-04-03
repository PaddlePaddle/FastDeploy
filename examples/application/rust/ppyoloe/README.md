English | [简体中文](README_CN.md)
# PaddleDetection Rust Deployment Example

This directory provides examples that `main.rs` and`build.rs` use `bindgen` to call FastDeploy C API and fast finish the deployment of PaddleDetection model PPYOLOE on CPU/GPU.

Before deployment, three steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)
- 2.  Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)
- 3. Download Rustup and install [Rust](https://www.rust-lang.org/tools/install)

Taking inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 1.0.4 above (x.x.x>1.0.4) or develop version (x.x.x=0.0.0) is required to support this model.

### Use Rust and bindgen to deploy PPYOLOE model

Download the FastDeploy precompiled library. Users can choose your appropriate version in the `FastDeploy Precompiled Library` mentioned above.
```bash
wget https://fastdeploy.bj.bcebos.com/dev/cpp/fastdeploy-linux-x64-0.0.0.tgz
tar xvf fastdeploy-linux-x64-0.0.0.tgz
```

Download the PPYOLOE model file and test images.
```bash
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
tar xvf ppyoloe_crn_l_300e_coco.tgz
```

In `build.rs`, configure the`cargo:rustc-link-search`to FastDeploy dynamic library path. The FastDeploy dynamic library is located in the `/lib` directory. Configure the`cargo:rustc-link-lib` to FastDeploy dynamic library`fastdeploy` and the`headers_dir`to FastDeploy C API directory path.
```bash
println!("cargo:rustc-link-search=./fastdeploy-linux-x64-0.0.0/lib");
println!("cargo:rustc-link-lib=fastdeploy");
let headers_dir = PathBuf::from("./fastdeploy-linux-x64-0.0.0/include");
```

Use the following command to add Fastdeploy library path to the environment variable.
```bash
source /Path/to/fastdeploy-linux-x64-0.0.0/fastdeploy_init.sh 
```

Use `Cargo` tool to compile the Rust project.
```bash
cargo build
```

After compiling, use the following command to obtain the predicted results.
```bash
# CPU inference
cargo run -- --model ./ppyoloe_crn_l_300e_coco --image 000000014439.jpg --device 0
# GPU inference
cargo run -- --model ./ppyoloe_crn_l_300e_coco --image 000000014439.jpg --device 1
```

Then visualized inspection result is saved in the local image `vis_result_ppyoloe.jpg`.
