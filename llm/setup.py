import setuptools

setuptools.setup(
    name="fastdeploy-llm",
    version="0.0.1",
    author="fastdeploy",
    author_email="fastdeploy@baidu.com",
    description="FastDeploy for Large Language Model",
    long_description_content_type="text/plain",
    url="https://github.com/PaddlePaddle/FastDeploy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license='Apache 2.0')
#    entry_points={'console_scripts': ['x2paddle=x2paddle.convert:main', ]})
