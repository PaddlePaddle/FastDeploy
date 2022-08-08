import setuptools
import fastdeploy
import io
import os

with open("requirements.txt") as fin:
    REQUIRED_PACKAGES = fin.read()


def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


setuptools.setup(
    name="fastdeploy-python",
    version=fastdeploy.__version__,
    author="FastDeploy",
    author_email="fastdeploy@baidu.com",
    description="FastDeploy is a toolkit to deploy deeplearning models.",
    long_description=read("../README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/PaddlePaddle/FastDeploy",
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PACKAGES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license='Apache 2.0',
    entry_points={
        'console_scripts': ['fastdeploy=fastdeploy.__init__:main', ]
    })
