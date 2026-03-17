@echo off
REM Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
REM
REM Licensed under the Apache License, Version 2.0 (the "License");
REM you may not use this file except in compliance with the License.
REM You may obtain a copy of the License at
REM
REM     http://www.apache.org/licenses/LICENSE-2.0
REM
REM Unless required by applicable law or agreed to in writing, software
REM distributed under the License is distributed on an "AS IS" BASIS,
REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
REM See the License for the specific language governing permissions and
REM limitations under the License.

REM FastDeploy Windows build script
REM Usage: build.bat [BUILD_MODE] [PYTHON]
REM   BUILD_MODE: 0 = build custom ops only, 1 = full build (default)
REM   PYTHON:     Python executable (default: python)
REM
REM Requires: Visual Studio Build Tools, CUDA Toolkit, Python 3.10+
REM Run from a Developer Command Prompt or ensure cl.exe is on PATH.

setlocal enabledelayedexpansion

set BUILD_MODE=%~1
if "%BUILD_MODE%"=="" set BUILD_MODE=1

set PYTHON=%~2
if "%PYTHON%"=="" set PYTHON=python

echo ============================================
echo  FastDeploy Windows Build
echo  Mode: %BUILD_MODE% (0=ops only, 1=full)
echo  Python: %PYTHON%
echo ============================================

REM Step 1: Build custom ops
echo.
echo [1] Building custom ops...
pushd custom_ops
%PYTHON% setup_ops.py install
if !ERRORLEVEL! neq 0 (
    echo [FAIL] Custom ops build failed.
    popd
    exit /b 1
)
popd
echo [OK] Custom ops built successfully.

if "%BUILD_MODE%"=="0" (
    echo.
    echo Build complete (ops only).
    exit /b 0
)

REM Step 2: Build and install FastDeploy wheel
echo.
echo [2] Building FastDeploy wheel...
%PYTHON% setup.py bdist_wheel
if !ERRORLEVEL! neq 0 (
    echo [FAIL] Wheel build failed.
    exit /b 1
)

echo [3] Installing FastDeploy wheel...
for %%w in (dist\fastdeploy*.whl) do (
    %PYTHON% -m pip install "%%w"
    if !ERRORLEVEL! neq 0 (
        echo [FAIL] pip install failed for %%w
        exit /b 1
    )
)

echo.
echo Build complete.
exit /b 0
