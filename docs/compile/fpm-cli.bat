@echo off
@rem --------------------------------------------------------------------------------------
@rem ACTION_TYPE: "help", "show", "install", "setup"
@rem      "help":         fpm-cli usage helps.
@rem      "init":         init FastDeploy path.
@rem      "show":         show all third libs need by FastDeploy.
@rem   "install":         install third libs to a specific location.
@rem     "setup":         setup FastDeploy third libs env for current terminal.
@rem INSTALL_TYPE: "dlls", "libs"
@rem      "dlls":         install DLLs files to a specific location.
@rem      "libs":         install LIBs files to a specific location.
@rem DESTINATION: specific location to install FastDeploy's third libs.
@rem      fpm-cli.bat help
@rem      fpm-cli.bat init path-to-fastdepploy-installed-dir
@rem      fpm-cli.bat show
@rem      fpm-cli.bat install dlls path-to-your-exe-or-dll-dir
@rem      fpm-cli.bat install libs path-to-your-lib-dir
@rem      fpm-cli.bat setup path-to-your-exe-or-dll-dir
@rem --------------------------------------------------------------------------------------

set ACTION_TYPE=%1
set INSTALL_TYPE=%2
set DESTINATION=%3
if "%ACTION_TYPE%" == "init" (
	set DESTINATION=%2
)
set FPM_INIT_PATH=fpm-init.txt
@rem Init FastDeploy package location.
@rem Usage: fpm-cli.bat init path-to-fastdepploy
if "%ACTION_TYPE%" == "init" (
	call:init_fpm_cli
	goto:eof
) else (
	set FASTDEPLOY_HOME=""
	if exist %FPM_INIT_PATH% (
		for /f %%i in (%FPM_INIT_PATH%) do (
			set FASTDEPLOY_HOME=%%i
		)
	) else (
	    echo [FastDeploy Package Manager][INFO] Can not find %FPM_INIT_PATH%
	    goto:eof
	)
)

@rem call funcs
call:help_fpm_cli  
call:show_packages
call:install_packages
call:setup_packages
call:clear_packages  
goto:eof

@rem "init_fpm_cli"
:init_fpm_cli
if exist %FPM_INIT_PATH% (
	del /Q %FPM_INIT_PATH%
)
if "%ACTION_TYPE%" == "init" (
	echo %DESTINATION%
	echo %DESTINATION% >>%FPM_INIT_PATH%
	echo [FastDeploy Package Manager][INFO] Init done: %FPM_INIT_PATH% [%DESTINATION%]
)
goto:eof
@rem end "init_fpm_cli"

@rem "help_fpm_cli" 
:help_fpm_cli
if "%ACTION_TYPE%" == "help" (
	echo ----------------------------------------------------------------------------------
	echo FASTDEPLOY_HOME: %FASTDEPLOY_HOME%
	echo ACTION_TYPE: "help", "show", "install", "setup"
	echo      "help":         fpm-cli usage helps.
	echo      "init":         init FastDeploy path.
	echo      "show":         show all third libs need by FastDeploy.
	echo   "install":         install third libs to a specific location.
	echo     "setup":         setup FastDeploy third libs env for current terminal.
	echo INSTALL_TYPE: "dlls", "libs"
	echo      "dlls":         install DLLs files to a specific location.
	echo      "libs":         install LIBs files to a specific location.
	echo DESTINATION: specific location to install FastDeploy's third libs.
	echo Usage: 
	echo      fpm-cli.bat help
	echo      fpm-cli.bat init path-to-fastdepploy-installed-dir
	echo      fpm-cli.bat show
	echo      fpm-cli.bat install dlls path-to-your-exe-or-dll-dir
	echo      fpm-cli.bat install libs path-to-your-exe-or-dll-dir
	echo      fpm-cli.bat setup path-to-your-exe-or-dll-dir
	echo ----------------------------------------------------------------------------------
)
goto:eof
@rem end "help_fpm_cli" 

@rem "show_packages"
:show_packages
if "%ACTION_TYPE%" == "show" (
	echo ----------------------------------------------------------------------------------
	echo [FastDeploy Package Manager][INFO] %ACTION_TYPE% FastDeploy third_libs ... 
	echo ---------------------------------- [DLLs][PATH] ----------------------------------
	echo %FASTDEPLOY_HOME%\lib\*.dll
	echo %FASTDEPLOY_HOME%\third_libs\install\onnxruntime\lib\*.dll
	echo %FASTDEPLOY_HOME%\third_libs\install\opencv-win-x64-3.4.16\build\x64\vc15\bin\*.dll
	echo %FASTDEPLOY_HOME%\third_libs\install\paddle_inference\paddle\lib\*.dll
	echo %FASTDEPLOY_HOME%\third_libs\install\paddle_inference\third_party\install\mkldnn\lib\*.dll
	echo %FASTDEPLOY_HOME%\third_libs\install\paddle_inference\third_party\install\mklml\lib\*.dll
	echo %FASTDEPLOY_HOME%\third_libs\install\paddle2onnx\lib\*.dll
	echo %FASTDEPLOY_HOME%\third_libs\install\tensorrt\lib\*.dll
	echo %FASTDEPLOY_HOME%\third_libs\install\faster_tokenizer\lib\*.dll
	echo %FASTDEPLOY_HOME%\third_libs\install\faster_tokenizer\third_party\lib\*.dll
	echo %FASTDEPLOY_HOME%\third_libs\install\yaml-cpp\lib\*.dll
	echo %FASTDEPLOY_HOME%\third_libs\install\openvino\bin\*.dll
	echo %FASTDEPLOY_HOME%\third_libs\install\openvino\3rdparty\tbb\bin\*.dll  
	echo ---------------------------------- [LIBs][PATH] ----------------------------------
	echo %FASTDEPLOY_HOME%\lib\*.lib
	echo %FASTDEPLOY_HOME%\third_libs\install\onnxruntime\lib\*.lib
	echo %FASTDEPLOY_HOME%\third_libs\install\opencv-win-x64-3.4.16\build\x64\vc15\lib\*.lib
	echo %FASTDEPLOY_HOME%\third_libs\install\paddle_inference\paddle\lib\*.lib
	echo %FASTDEPLOY_HOME%\third_libs\install\paddle_inference\third_party\install\mkldnn\lib\*.lib
	echo %FASTDEPLOY_HOME%\third_libs\install\paddle_inference\third_party\install\mklml\lib\*.lib
	echo %FASTDEPLOY_HOME%\third_libs\install\paddle2onnx\lib\*.lib
	echo %FASTDEPLOY_HOME%\third_libs\install\tensorrt\lib\*.lib
	echo %FASTDEPLOY_HOME%\third_libs\install\faster_tokenizer\lib\*.lib
	echo %FASTDEPLOY_HOME%\third_libs\install\faster_tokenizer\third_party\lib\*.lib
	echo %FASTDEPLOY_HOME%\third_libs\install\yaml-cpp\lib\*.lib
	echo %FASTDEPLOY_HOME%\third_libs\install\openvino\lib\*.lib
	echo %FASTDEPLOY_HOME%\third_libs\install\openvino\3rdparty\tbb\lib\*.lib  
	echo ----------------------------------------------------------------------------------
) 
goto:eof
@rem end "show_packages"


@rem "install_packages" 
:install_packages
if "%ACTION_TYPE%" == "install" (
	echo [FastDeploy Package Manager][INFO] %ACTION_TYPE% FastDeploy third_libs ... 
) else ( goto:eof )
if not exist %DESTINATION% (goto:eof)
if "%INSTALL_TYPE%" == "dlls" (
	copy /Y %FASTDEPLOY_HOME%\lib\*.dll %DESTINATION%
	copy /Y %FASTDEPLOY_HOME%\third_libs\install\onnxruntime\lib\*.dll %DESTINATION%
	copy /Y %FASTDEPLOY_HOME%\third_libs\install\opencv-win-x64-3.4.16\build\x64\vc15\bin\*.dll %DESTINATION%
	copy /Y %FASTDEPLOY_HOME%\third_libs\install\paddle_inference\paddle\lib\*.dll %DESTINATION%
	copy /Y %FASTDEPLOY_HOME%\third_libs\install\paddle_inference\third_party\install\mkldnn\lib\*.dll %DESTINATION%
	copy /Y %FASTDEPLOY_HOME%\third_libs\install\paddle_inference\third_party\install\mklml\lib\*.dll %DESTINATION%
	copy /Y %FASTDEPLOY_HOME%\third_libs\install\paddle2onnx\lib\*.dll %DESTINATION%
	copy /Y %FASTDEPLOY_HOME%\third_libs\install\tensorrt\lib\*.dll %DESTINATION%
	copy /Y %FASTDEPLOY_HOME%\third_libs\install\faster_tokenizer\lib\*.dll %DESTINATION%
	copy /Y %FASTDEPLOY_HOME%\third_libs\install\faster_tokenizer\third_party\lib\*.dll %DESTINATION%
	copy /Y %FASTDEPLOY_HOME%\third_libs\install\yaml-cpp\lib\*.dll %DESTINATION%
	copy /Y %FASTDEPLOY_HOME%\third_libs\install\openvino\bin\*.dll %DESTINATION%
	copy /Y %FASTDEPLOY_HOME%\third_libs\install\openvino\bin\*.xml %DESTINATION%
	copy /Y %FASTDEPLOY_HOME%\third_libs\install\openvino\3rdparty\tbb\bin\*.dll %DESTINATION%
	echo [FastDeploy Package Manager][INFO] Installed DLLs to: %DESTINATION% 
)
if "%INSTALL_TYPE%" == "libs" (
	copy /Y  %FASTDEPLOY_HOME%\lib\*.lib %DESTINATION%
	copy /Y  %FASTDEPLOY_HOME%\third_libs\install\onnxruntime\lib\*.lib %DESTINATION%
	copy /Y  %FASTDEPLOY_HOME%\third_libs\install\opencv-win-x64-3.4.16\build\x64\vc15\lib\*.lib %DESTINATION%
	copy /Y  %FASTDEPLOY_HOME%\third_libs\install\paddle_inference\paddle\lib\*.lib %DESTINATION%
	copy /Y  %FASTDEPLOY_HOME%\third_libs\install\paddle_inference\third_party\install\mkldnn\lib\*.lib %DESTINATION%
	copy /Y  %FASTDEPLOY_HOME%\third_libs\install\paddle_inference\third_party\install\mklml\lib\*.lib %DESTINATION%
	copy /Y  %FASTDEPLOY_HOME%\third_libs\install\paddle2onnx\lib\*.lib %DESTINATION%
	copy /Y  %FASTDEPLOY_HOME%\third_libs\install\tensorrt\lib\*.lib %DESTINATION%
	copy /Y  %FASTDEPLOY_HOME%\third_libs\install\faster_tokenizer\lib\*.lib %DESTINATION%
	copy /Y  %FASTDEPLOY_HOME%\third_libs\install\faster_tokenizer\third_party\lib\*.lib %DESTINATION%
	copy /Y  %FASTDEPLOY_HOME%\third_libs\install\yaml-cpp\lib\*.lib %DESTINATION%
	copy /Y  %FASTDEPLOY_HOME%\third_libs\install\openvino\lib\*.lib %DESTINATION%
	copy /Y  %FASTDEPLOY_HOME%\third_libs\install\openvino\3rdparty\tbb\lib\*.lib %DESTINATION%
	echo [FastDeploy Package Manager][INFO] Installed LIBs to: %DESTINATION% 
)
goto:eof
@rem end "install_packages"


@rem "setup_packages" TODO(qiuyanjun)
:setup_packages 
if "%ACTION_TYPE%" == "setup" ( 
	echo [FastDeploy Package Manager][INFO] %ACTION_TYPE% FastDeploy third_libs ... 
) else ( goto:eof )
set PATH=%FASTDEPLOY_HOME%\lib;%PATH%
set PATH=%FASTDEPLOY_HOME%\lib;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\onnxruntime\lib;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\opencv-win-x64-3.4.16\build\x64\vc15\bin;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\paddle_inference\paddle\lib;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\paddle_inference\third_party\install\mkldnn\lib;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\paddle_inference\third_party\install\mklml\lib;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\paddle2onnx\lib;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\tensorrt\lib;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\faster_tokenizer\lib;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\faster_tokenizer\third_party\lib;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\yaml-cpp\lib;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\openvino\bin;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\openvino\3rdparty\tbb\bin;%PATH%
if exist %DESTINATION% (
	copy /Y %FASTDEPLOY_HOME%\third_libs\install\onnxruntime\lib\*.dll %DESTINATION%
)
goto:eof 
@rem end "setup_packages"


@rem "clear_packages"
:clear_packages  
if "%ACTION_TYPE%" == "clear" (
	echo [FastDeploy Package Manager][ERROR] Not support %ACTION_TYPE% FastDeploy third_libs action now ... 
)
goto:eof
@rem end "clear_packages"
