#!/bin/bash

set -o errexit
set -o nounset
shopt -s extglob



if [ $# == 0 ];then
    ENV="test"
else
    ENV=$1
fi

# clear dist
rm -rf dist

# build
python3 setup.py sdist bdist_wheel


# publish

if [ ${ENV} == 'production' ];then
    echo "publish to https://pypi.org"
    python3 -m twine upload dist/*
else
    echo "publish to https://test.pypi.org/"
    python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
fi

# del egg-info and build
rm -rf paddlejsconverter.egg-info
rm -rf build

