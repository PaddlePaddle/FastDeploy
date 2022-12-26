English | [中文](../../cn/faq/use_sdk_on_linux.md)

# C++ Deployment on Linux

1. After compilation, and execute the binary file, throw error `error while loading shared libraries`

When we execute the binary file, it requires the dependent libraries can be found in the system, otherwise, it throws error like below
```
./infer_ppyoloe_demo: error while loading shared libraries: libonnxruntime.so.1.12.0: cannot open shared object file: No such file or directory
```

FastDeploy provides a shell scripts to help export the libraries path to `LD_LIBRARY_PATH`, execute the following command

```
source /Downloads/fastdeploy-linux-x64-1.0.0/fastdeploy_init.sh
```

And now you can execute the binary file again. 
