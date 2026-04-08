# collect-env: Environment Information Collector

`collect-env` is used to gather information about the system, GPU, deep learning frameworks, and FastDeploy environment.
This subcommand requires no additional arguments — simply run it directly.

## Usage

```
fastdeploy collect-env
```

## Information Collected

**1. System Information**

* `os`: Operating system

  * Linux: `lsb_release -a` or `cat /etc/*-release`
  * Windows: `wmic os get Caption`
  * macOS: `sw_vers -productVersion`
* `gcc_version`: GCC version, retrieved by `gcc --version`
* `clang_version`: Clang version, retrieved by `clang --version`
* `cmake_version`: CMake version, retrieved by `cmake --version`
* `libc_version`: GNU C library version (Linux only), retrieved by `platform.libc_ver()`

**2. PyTorch Information**

* `torch_version`: PyTorch version
* `is_debug_build`: Whether it’s a Debug build
* `cuda_compiled_version`: CUDA version used to compile PyTorch
* `hip_compiled_version`: HIP version used to compile PyTorch (for AMD GPUs)

**3. Paddle Information**

* `paddle_version`: Paddle version
* `paddle_compiled_version`: CUDA version used to compile Paddle

**4. Python Environment**

* `python_version`: Python version
* `python_platform`: Detailed platform information

**5. CUDA / GPU Information**

* `is_cuda_available`: Whether CUDA is available
* `cuda_runtime_version`: CUDA runtime version
* `cuda_module_loading`: CUDA module loading policy (`CUDA_MODULE_LOADING` environment variable)
* `nvidia_gpu_models`: GPU model(s)
* `nvidia_driver_version`: NVIDIA driver version
* `cudnn_version`: cuDNN version
* `caching_allocator_config`: CUDA caching allocator configuration (`PYTORCH_CUDA_ALLOC_CONF` environment variable)
* `is_xnnpack_available`: Whether XNNPACK is available

**6. CPU Information**

* `cpu_info`: Detailed CPU information (retrieved via `lscpu` or Windows system commands)

**7. Relevant Library Versions**

* `pip_packages`: Key library versions collected via `python -m pip list --format=freeze`
* `conda_packages`: Key library versions collected via `conda list`

**8. FastDeploy-Specific Information**

* `fastdeploy_version`: FastDeploy version (development builds include Git commit hash)
* `fastdeploy_build_flags`: Build flags (e.g., targeted CUDA architectures from `FD_BUILDING_ARCS`)
* `gpu_topo`: GPU topology (retrieved via `nvidia-smi topo -m`)

**9. Environment Variables**

* `env_vars`: Environment variables starting with `TORCH`, `CUDA`, `NCCL`, or FastDeploy-specific prefixes

  * Sensitive variables containing `secret`, `token`, etc., are filtered out.

---

## Example Output
```
==============================
        System Info
==============================
OS                           : Ubuntu 20.04.6 LTS (x86_64)
GCC version                  : (GCC) 12.2.0
Clang version                : 3.8.0 (tags/RELEASE_380/final)
CMake version                : version 3.18.0
Libc version                 : glibc-2.31

==============================
       PyTorch Info
==============================
PyTorch version              : 2.5.1+cu118
Is debug build               : False
CUDA used to build PyTorch   : 11.8

==============================
       Paddle Info
==============================
Paddle version              : 3.1.0
CUDA used to build paddle   : 12.6

==============================
      Python Environment
==============================
Python version               : 3.10.16 (main, Dec 11 2024, 16:24:50) [GCC 11.2.0] (64-bit runtime)
Python platform              : Linux-5.10.0-1.0.0.28-x86_64-with-glibc2.31

==============================
       CUDA / GPU Info
==============================
Is CUDA available            : True
CUDA runtime version         : 12.3.103
CUDA_MODULE_LOADING set to   : LAZY
GPU models and configuration :
GPU 0: NVIDIA A100-SXM4-40GB
GPU 1: NVIDIA A100-SXM4-40GB
GPU 2: NVIDIA A100-SXM4-40GB
GPU 3: NVIDIA A100-SXM4-40GB
GPU 4: NVIDIA A100-SXM4-40GB
GPU 5: NVIDIA A100-SXM4-40GB
GPU 6: NVIDIA A100-SXM4-40GB
GPU 7: NVIDIA A100-SXM4-40GB

Nvidia driver version        : 525.125.06
cuDNN version                : Could not collect
Is XNNPACK available         : True

==============================
          CPU Info
==============================
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Byte Order:                      Little Endian
Address sizes:                   46 bits physical, 48 bits virtual
CPU(s):                          160
On-line CPU(s) list:             0-159
Thread(s) per core:              2
Core(s) per socket:              20
Socket(s):                       4
NUMA node(s):                    4
Vendor ID:                       GenuineIntel
CPU family:                      6
Model:                           85
Model name:                      Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz
Stepping:                        7
CPU MHz:                         3199.750
CPU max MHz:                     3900.0000
CPU min MHz:                     1000.0000
BogoMIPS:                        5000.00
Virtualization:                  VT-x
L1d cache:                       2.5 MiB
L1i cache:                       2.5 MiB
L2 cache:                        80 MiB
L3 cache:                        110 MiB
NUMA node0 CPU(s):               0-19,80-99
NUMA node1 CPU(s):               20-39,100-119
NUMA node2 CPU(s):               40-59,120-139
NUMA node3 CPU(s):               60-79,140-159
Vulnerability Itlb multihit:     KVM: Mitigation: VMX disabled
Vulnerability L1tf:              Not affected
Vulnerability Mds:               Not affected
Vulnerability Meltdown:          Not affected
Vulnerability Spec store bypass: Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:        Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:        Mitigation; Enhanced IBRS, IBPB conditional, RSB filling
Vulnerability Srbds:             Not affected
Vulnerability Tsx async abort:   Mitigation; TSX disabled
Flags:                           fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cdp_l3 invpcid_single intel_ppin ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm mpx rdt_a avx512f avx512dq rdseed adx smap clflushopt clwb intel_pt avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local dtherm ida arat pln pts pku avx512_vnni md_clear flush_l1d arch_capabilities

==============================
Versions of relevant libraries
==============================
[pip3] aiozmq==1.0.0
[pip3] flake8==7.2.0
[pip3] numpy==1.26.4
[pip3] nvidia-cublas-cu11==11.11.3.6
[pip3] nvidia-cublas-cu12==12.6.4.1
[pip3] nvidia-cuda-cccl-cu12==12.6.77
[pip3] nvidia-cuda-cupti-cu11==11.8.87
[pip3] nvidia-cuda-cupti-cu12==12.6.80
[pip3] nvidia-cuda-nvrtc-cu11==11.8.89
[pip3] nvidia-cuda-nvrtc-cu12==12.6.77
[pip3] nvidia-cuda-runtime-cu11==11.8.89
[pip3] nvidia-cuda-runtime-cu12==12.6.77
[pip3] nvidia-cudnn-cu11==9.1.0.70
[pip3] nvidia-cudnn-cu12==9.5.1.17
[pip3] nvidia-cufft-cu11==10.9.0.58
[pip3] nvidia-cufft-cu12==11.3.0.4
[pip3] nvidia-cufile-cu12==1.11.1.6
[pip3] nvidia-curand-cu11==10.3.0.86
[pip3] nvidia-curand-cu12==10.3.7.77
[pip3] nvidia-cusolver-cu11==11.4.1.48
[pip3] nvidia-cusolver-cu12==11.7.1.2
[pip3] nvidia-cusparse-cu11==11.7.5.86
[pip3] nvidia-cusparse-cu12==12.5.4.2
[pip3] nvidia-cusparselt-cu12==0.6.3
[pip3] nvidia-ml-py==12.575.51
[pip3] nvidia-nccl-cu11==2.21.5
[pip3] nvidia-nccl-cu12==2.25.1
[pip3] nvidia-nvjitlink-cu12==12.6.85
[pip3] nvidia-nvtx-cu11==11.8.86
[pip3] nvidia-nvtx-cu12==12.6.77
[pip3] onnx==1.18.0
[pip3] onnxoptimizer==0.3.13
[pip3] paddle2onnx==2.0.1
[pip3] pynvml==12.0.0
[pip3] pyzmq==26.4.0
[pip3] torch==2.5.1+cu118
[pip3] torchaudio==2.5.1+cu118
[pip3] torchvision==0.20.1+cu118
[pip3] transformers==4.55.4
[pip3] triton==3.3.0
[pip3] use_triton_in_paddle==0.1.0
[pip3] zmq==0.0.0
[conda] aiozmq                    1.0.0                    pypi_0    pypi
[conda] numpy                     1.26.4                   pypi_0    pypi
[conda] nvidia-cublas-cu11        11.11.3.6                pypi_0    pypi
[conda] nvidia-cublas-cu12        12.6.4.1                 pypi_0    pypi
[conda] nvidia-cuda-cccl-cu12     12.6.77                  pypi_0    pypi
[conda] nvidia-cuda-cupti-cu11    11.8.87                  pypi_0    pypi
[conda] nvidia-cuda-cupti-cu12    12.6.80                  pypi_0    pypi
[conda] nvidia-cuda-nvrtc-cu11    11.8.89                  pypi_0    pypi
[conda] nvidia-cuda-nvrtc-cu12    12.6.77                  pypi_0    pypi
[conda] nvidia-cuda-runtime-cu11  11.8.89                  pypi_0    pypi
[conda] nvidia-cuda-runtime-cu12  12.6.77                  pypi_0    pypi
[conda] nvidia-cudnn-cu11         9.1.0.70                 pypi_0    pypi
[conda] nvidia-cudnn-cu12         9.5.1.17                 pypi_0    pypi
[conda] nvidia-cufft-cu11         10.9.0.58                pypi_0    pypi
[conda] nvidia-cufft-cu12         11.3.0.4                 pypi_0    pypi
[conda] nvidia-cufile-cu12        1.11.1.6                 pypi_0    pypi
[conda] nvidia-curand-cu11        10.3.0.86                pypi_0    pypi
[conda] nvidia-curand-cu12        10.3.7.77                pypi_0    pypi
[conda] nvidia-cusolver-cu11      11.4.1.48                pypi_0    pypi
[conda] nvidia-cusolver-cu12      11.7.1.2                 pypi_0    pypi
[conda] nvidia-cusparse-cu11      11.7.5.86                pypi_0    pypi
[conda] nvidia-cusparse-cu12      12.5.4.2                 pypi_0    pypi
[conda] nvidia-cusparselt-cu12    0.6.3                    pypi_0    pypi
[conda] nvidia-ml-py              12.575.51                pypi_0    pypi
[conda] nvidia-nccl-cu11          2.21.5                   pypi_0    pypi
[conda] nvidia-nccl-cu12          2.25.1                   pypi_0    pypi
[conda] nvidia-nvjitlink-cu12     12.6.85                  pypi_0    pypi
[conda] nvidia-nvtx-cu11          11.8.86                  pypi_0    pypi
[conda] nvidia-nvtx-cu12          12.6.77                  pypi_0    pypi
[conda] pynvml                    12.0.0                   pypi_0    pypi
[conda] pyzmq                     26.4.0                   pypi_0    pypi
[conda] torch                     2.5.1+cu118              pypi_0    pypi
[conda] torchaudio                2.5.1+cu118              pypi_0    pypi
[conda] torchvision               0.20.1+cu118             pypi_0    pypi
[conda] transformers              4.55.4                   pypi_0    pypi
[conda] triton                    3.3.0                    pypi_0    pypi
[conda] use-triton-in-paddle      0.1.0                    pypi_0    pypi
[conda] zmq                       0.0.0                    pypi_0    pypi

==============================
         FastDeploy Info
==============================
FastDeply Version                 : 2.0.0a0
FastDeply Build Flags:
  CUDA Archs: [];
GPU Topology:
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    NIC0    NIC1    NIC2    CPU Affinity    NUMA Affinity
GPU0     X      NV12    NV12    NV12    NV12    NV12    NV12    NV12    PXB     SYS     SYS     0-19,80-99      0
GPU1    NV12     X      NV12    NV12    NV12    NV12    NV12    NV12    PXB     SYS     SYS     0-19,80-99      0
GPU2    NV12    NV12     X      NV12    NV12    NV12    NV12    NV12    SYS     NODE    PXB     20-39,100-119   1
GPU3    NV12    NV12    NV12     X      NV12    NV12    NV12    NV12    SYS     NODE    PXB     20-39,100-119   1
GPU4    NV12    NV12    NV12    NV12     X      NV12    NV12    NV12    SYS     SYS     SYS     40-59,120-139   2
GPU5    NV12    NV12    NV12    NV12    NV12     X      NV12    NV12    SYS     SYS     SYS     40-59,120-139   2
GPU6    NV12    NV12    NV12    NV12    NV12    NV12     X      NV12    SYS     SYS     SYS     60-79,140-159   3
GPU7    NV12    NV12    NV12    NV12    NV12    NV12    NV12     X      SYS     SYS     SYS     60-79,140-159   3
NIC0    PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS      X      SYS     SYS
NIC1    SYS     SYS     NODE    NODE    SYS     SYS     SYS     SYS     SYS      X      NODE
NIC2    SYS     SYS     PXB     PXB     SYS     SYS     SYS     SYS     SYS     NODE     X

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
  NIC2: mlx5_2

==============================
     Environment Variables
==============================
NVIDIA_VISIBLE_DEVICES=GPU-0fe14fa3-b286-3d79-b223-1912257b4d64,GPU-282b567f-d2c4-f472-5c0d-975a7d96e1a7,GPU-a9d7e24d-1bb2-eb83-63fb-40584754f4be,GPU-924f3dc2-1b05-c35d-12f5-53d9458a1bd2,GPU-57591c1d-c444-18b8-c29d-f44cbaae8142,GPU-a28a9121-042a-81cf-d759-83ce1e3b962a,GPU-c124b75e-2768-6b7d-41fa-46dbf0159c87,GPU-b196a47d-c21e-1ec3-8003-5d776173ec7c
NCCL_P2P_DISABLE=0
NVIDIA_REQUIRE_CUDA=cuda>=12.3 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471 brand=tesla,driver>=525,driver<526 brand=unknown,driver>=525,driver<526 brand=nvidia,driver>=525,driver<526 brand=nvidiartx,driver>=525,driver<526 brand=geforce,driver>=525,driver<526 brand=geforcertx,driver>=525,driver<526 brand=quadro,driver>=525,driver<526 brand=quadrortx,driver>=525,driver<526 brand=titan,driver>=525,driver<526 brand=titanrtx,driver>=525,driver<526 brand=tesla,driver>=535,driver<536 brand=unknown,driver>=535,driver<536 brand=nvidia,driver>=535,driver<536 brand=nvidiartx,driver>=535,driver<536 brand=geforce,driver>=535,driver<536 brand=geforcertx,driver>=535,driver<536 brand=quadro,driver>=535,driver<536 brand=quadrortx,driver>=535,driver<536 brand=titan,driver>=535,driver<536 brand=titanrtx,driver>=535,driver<536
NCCL_IB_CUDA_SUPPORT=0
NVIDIA_LIB=/usr/local/nvidia/lib64
NCCL_VERSION=2.19.3-1
NCCL_SOCKET_IFNAME=xgbe1
NVIDIA_GDRCOPY=enabled
NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH
NVIDIA_DRIVER_CAPABILITIES=compute,utility
NCCL_DEBUG=INFO
NCCL_LIBRARY_PATH=/usr/local/nccl
NVIDIA_VISIBLE_GPUS_UUID=GPU-0fe14fa3-b286-3d79-b223-1912257b4d64,GPU-282b567f-d2c4-f472-5c0d-975a7d96e1a7,GPU-a9d7e24d-1bb2-eb83-63fb-40584754f4be,GPU-924f3dc2-1b05-c35d-12f5-53d9458a1bd2,GPU-57591c1d-c444-18b8-c29d-f44cbaae8142,GPU-a28a9121-042a-81cf-d759-83ce1e3b962a,GPU-c124b75e-2768-6b7d-41fa-46dbf0159c87,GPU-b196a47d-c21e-1ec3-8003-5d776173ec7c
NVIDIA_PRODUCT_NAME=CUDA
NCCL_IB_GID_INDEX=3
CUDA_VERSION=12.3.1
NVIDIA_TOOLS=/home/opt/cuda_tools
NCCL_DEBUG_FILE=/root/paddlejob/workspace/log/nccl.%h.%p.log
NCCL_IB_QPS_PER_CONNECTION=2
NCCL_IB_CONNECT_RETRY_CNT=15
NCCL_ERROR_FILE=/root/paddlejob/workspace/log/err.%h.%p.log
NCCL_IB_TIMEOUT=22
CUDNN_VERSION=9.0.0
NCCL_IB_DISABLE=0
NVIDIA_VISIBLE_GPUS_SLOT=6,7,0,1,2,3,4,5
NCCL_IB_ADAPTIVE_ROUTING=1
OMP_NUM_THREADS=1
CUDA_MODULE_LOADING=LAZY
```
