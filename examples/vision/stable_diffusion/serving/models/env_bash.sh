#!/bin/sh
sex -x
ln -s /usr/bin/python3 /usr/bin/python;
pip uninstall paddlenlp;
pip install ppdiffusers;
pip install paddlepaddle-gpu==0.0.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html;
pip install fastdeploy-gpu-python==0.0.0 -f https://www.paddlepaddle.org.cn/whl/fastdeploy_nightly_build.html;
pip install tritonclient[all];
python export_model.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --output_path stable-diffusion-v1-5 --height=512 --width=512 && mv stable-diffusion-v1-5 ./stable_diffusion/1/stable-diffusion-v1-5;
