import setuptools

long_description = "fastdeploy-tools is a toolkit for FastDeploy, including auto compression .etc.\n\n"
long_description += "Usage of auto compression: fastdeploy compress --config_path=./yolov7_tiny_qat_dis.yaml --method='QAT' --save_dir='./v7_qat_outmodel/' \n"

install_requires = ['uvicorn==0.16.0']

setuptools.setup(
    name="fastdeploy-tools",  # name of package
    version="0.0.3",  #version of package
    description="A toolkit for FastDeploy.",
    long_description=long_description,
    long_description_content_type="text/plain",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license='Apache 2.0',
    entry_points={
        'console_scripts': ['fastdeploy = common_tools.common_tools:main', ]
    })
