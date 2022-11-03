import setuptools
import fd_auto_compress

long_description = "fastdeploy-auto-compression is a toolkit for model auto compression of FastDeploy.\n\n"
long_description += "Usage: fastdeploy_auto_compress --config_path=./yolov7_tiny_qat_dis.yaml --method='QAT' --save_dir='../v7_qat_outmodel/' \n"

with open("requirements.txt") as fin:
    REQUIRED_PACKAGES = fin.read()

setuptools.setup(
    name="fastdeploy-auto-compression",  # name of package
    description="A toolkit for model auto compression of FastDeploy.",
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
        'console_scripts':
        ['fastdeploy_auto_compress=fd_auto_compress.fd_auto_compress:main', ]
    })
