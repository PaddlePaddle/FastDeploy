import setuptools
import fdquant

long_description = "FDQuant is a toolkit for model quantization of FastDeploy.\n\n"
long_description += "Usage: fastdeploy_quant --config_path=./yolov7_tiny_qat_dis.yaml --method='QAT' --save_dir='../v7_qat_outmodel/' \n"

with open("requirements.txt") as fin:
    REQUIRED_PACKAGES = fin.read()

setuptools.setup(
    name="fastdeploy-quantization",  # name of package
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
    entry_points={
        'console_scripts': ['fastdeploy_quant=fdquant.fdquant:main', ]
    })
