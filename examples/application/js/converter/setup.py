import setuptools

PY_MODILES = ["convertToPaddleJSModel", "convertModel", "optimizeModel", "pruningModel", "rnn", "fuseOps"]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="paddlejsconverter",
    version="1.0.7",
    author="paddlejs",
    author_email="382248373@qq.com",
    description="Paddlejs model converter",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PaddlePaddle/Paddle.js",
    py_modules=PY_MODILES,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "paddlepaddle >= 2.0.0",
        "paddlejslite >= 0.0.2",
        "numpy"
    ],
    entry_points={
        "console_scripts": [
            "paddlejsconverter = convertToPaddleJSModel:main",
            "pdjsConvertModel = convertModel:main",
            "pdjsOptimizeModel = optimizeModel:main"
        ]
    }
)
