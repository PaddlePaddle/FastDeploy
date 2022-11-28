# Copyright (c) 2022 Baidu, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A setup module for the poros Python package.
"""

# for python2 compatiblity
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import sys
import setuptools
from setuptools import find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
import shutil
from torch.utils import cpp_extension
from wheel.bdist_wheel import bdist_wheel
from distutils.cmd import Command
from distutils import spawn
from distutils.sysconfig import get_python_lib
import multiprocessing

# Constant known variables used throughout this file
THREAD_NUM = multiprocessing.cpu_count()
CXX11_ABI = False
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

if "--use-cxx11-abi" in sys.argv:
    sys.argv.remove("--use-cxx11-abi")
    CXX11_ABI = True

def cmake_build():
    """execute cmake build, to make the shared lib `libporos.so` """
    cwd = os.getcwd()
    if spawn.find_executable('cmake') is None:
        sys.stderr.write("CMake is required to build this package.\n")
        sys.exit(-1)
    _source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _build_dir = os.path.join(_source_dir, 'build')
    _prefix = get_python_lib()
    try:
        cmake_configure_command = [
            'cmake',
            '-H{0}'.format(_source_dir),
            '-B{0}'.format(_build_dir),
            '-DCMAKE_INSTALL_PREFIX={0}'.format(_prefix),
        ]
        _generator = os.getenv('CMAKE_GENERATOR')
        if _generator is not None:
            cmake_configure_command.append('-G{0}'.format(_generator))
        spawn.spawn(cmake_configure_command)
        spawn.spawn(
            ['cmake', '--build', _build_dir, '-j', str(THREAD_NUM)])
        os.chdir(cwd)
    except spawn.DistutilsExecError:
        sys.stderr.write("Error while building with CMake\n")
        sys.exit(-1)

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    PY_CLEAN_FILES = [
        './build', './dist', './poros/__pycache__', './poros/lib', './*.pyc', './*.tgz', './*.egg-info'
    ]
    description = "Command to tidy up the project root"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for path_spec in self.PY_CLEAN_FILES:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(os.path.normpath(os.path.join(CURRENT_PATH, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(CURRENT_PATH):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" % (path, CURRENT_PATH))
                print('Removing %s' % os.path.relpath(path))
                shutil.rmtree(path)


if __name__ == "__main__":
    """main setup function"""
    poros_lib_path = os.path.join(CURRENT_PATH, "poros", "lib")

    # build libporos.so
    if "clean" not in sys.argv:
        cmake_build()

    if os.path.exists('./poros/lib') == False:
        os.mkdir('./poros/lib')

    shutil.copy("../build/lib/libporos.so", "./poros/lib/libporos.so")

    # this is for torch customer extension
    C = cpp_extension.CppExtension(
        'poros._C', [
            'poros/csrc/poros_py.cpp',
        ],
        library_dirs=[poros_lib_path, "../third_party/tensorrtlib"],
        libraries=["poros"],
        include_dirs=[
            CURRENT_PATH + "/poros/csrc",
            CURRENT_PATH + "/../build/include",

        ],
        extra_compile_args=[
            "-Wno-deprecated",
            "-Wno-deprecated-declarations",
            "-Wno-unused-function",
            '-Werror',
            '-fopenmp',
            '-D__const__=', '-g', '-O2', '-fPIC',
        ],
        extra_link_args=[
            "-Wno-deprecated", "-Wno-deprecated-declarations",
            "-Wno-unused-function",
            "-Wl,--no-as-needed",
            "-lporos",
            "-lnvinfer",
            "-lnvinfer_plugin",
            "-Wl,-rpath,$ORIGIN/lib",
            "-lpthread", "-ldl", "-lutil", "-lrt", "-lm", "-Xlinker", "-export-dynamic",
        ],
    )

    setuptools.setup(
        name="poros",
        version="0.1.0",
        author="PorosTeam@BaiDu",
        description='A compiler backend for PyTorch and automatically accelerate inference using tensorrt engine',
        ext_modules=[C],
        packages=find_packages(),
        include_package_data=True,
        package_data={
            'poros': ['lib/*.so'],
        },
        exclude_package_data={
            '': ['*.cpp', '*.h'],
            'poros': ['csrc/*.cpp'],
        },
        install_requires=[
            'torch>=1.9.0',
        ],
        cmdclass={
            'clean': CleanCommand,
            'build_ext': cpp_extension.BuildExtension,
        },
    )
    print("Setup baidu.poros python module success!")
