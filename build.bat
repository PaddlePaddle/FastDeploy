REM Hackathon 10th Spring No.46 — Windows build support
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
REM Requires: Visual Studio Build Tools, CUDA Toolkit, Python 3.10+
REM Run from a Developer Command Prompt or ensure cl.exe is on PATH.

setlocal enabledelayedexpansion

if "%~1"=="/?" goto :show_help
if "%~1"=="-h" goto :show_help
if "%~1"=="--help" goto :show_help

set BUILD_MODE=%~1
if "%BUILD_MODE%"=="" set BUILD_MODE=1

set PYTHON=%~2
if "%PYTHON%"=="" set PYTHON=python

REM Python version check (require >= 3.9)
for /f "tokens=2 delims= " %%v in ('%PYTHON% --version 2^>^&1') do set PY_VER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do (
    if %%a LSS 3 (
        echo [FAIL] Python 3.9+ required, found %PY_VER%
        exit /b 1
    )
    if %%a==3 if %%b LSS 9 (
        echo [FAIL] Python 3.9+ required, found %PY_VER%
        exit /b 1
    )
)

echo ============================================
echo  FastDeploy Windows Build
echo  Mode: %BUILD_MODE% (0=ops only, 1=full)
echo  Python: %PYTHON% (%PY_VER%)
if defined FD_BUILDING_ARCS echo  CUDA Arcs: %FD_BUILDING_ARCS%
if defined FD_CPU_USE_BF16 echo  CPU BF16: %FD_CPU_USE_BF16%
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

:show_help
echo Usage: build.bat [BUILD_MODE] [PYTHON]
echo.
echo BUILD_MODE modes:
echo   0  Build custom ops only (no wheel packaging or pip install)
echo   1  Full build: compile C++ ops + build wheel + pip install (default)
echo.
echo Arguments:
echo   PYTHON  Python executable (default: python)
echo.
echo Environment variables:
echo   FD_BUILDING_ARCS    Target CUDA architectures, e.g. "[80, 90, 100]"
echo   FD_CPU_USE_BF16     Enable CPU BF16 ops: true/false
echo.
echo Examples:
echo   build.bat 1 python             Full build with default Python
echo   set FD_BUILDING_ARCS=[90]
echo   build.bat 0                    Build ops only for SM90
exit /b 0
