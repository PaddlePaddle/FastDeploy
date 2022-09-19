import setuptools
import fdquant

long_description = "FDQuant is a toolkit for model quantization of FastDeploy.\n\n"
long_description += "Usage: FDQuant --model_type YOLOV5 --model_file yolov5s.onnx --save_dir quant_out --data_dir=./COCO_train \n"

with open("requirements.txt") as fin:
    REQUIRED_PACKAGES = fin.read()

setuptools.setup(
    name="fdquant",
    description="A toolkit for model quantization of FastDeploy.",
    long_description=long_description,
    long_description_content_type="text/plain",
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PACKAGES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license='Apache 2.0',
    entry_points={'console_scripts': ['fdquant=fdquant.fdquant:main', ]})
