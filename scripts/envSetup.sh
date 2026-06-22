export MACA_PATH=/opt/maca

if [ ! -d ${HOME}/cu-bridge ]; then
  `${MACA_PATH}/tools/cu-bridge/tools/pre_make`
fi

export CUCC_PATH=/opt/maca/tools/cu-bridge
export CUCC_CMAKE_ENTRY=2
export CUDA_PATH=${HOME}/cu-bridge/CUDA_DIR
export PATH=${CUDA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:$LD_LIBRARY_PATH
export MACA_VISIBLE_DEVICES="0"
export PADDLE_XCCL_BACKEND=metax_gpu
export FLAGS_weight_only_linear_arch=80
export FD_MOE_BACKEND=cutlass
export ENABLE_V1_KVCACHE_SCHEDULER=1
export FD_ENC_DEC_BLOCK_NUM=2
export FD_SAMPLING_CLASS="rejection"   # 受编译器升级到 llvm19 影响，top_p_sampling 算子会出现阻塞情况

export PYTHONPATH="/data/FastDeploy:${PYTHONPATH:-}"
if [ ! -f /tmp/shm_redirect.so ]; then
    cat > /tmp/shm_redirect.c << 'CEOF'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <errno.h>
static const char* REDIRECT_DIR = "/tmp/shm";
static char* redirect_path(const char* name) {
    static __thread char buf[512];
    if (name[0] == '/') name++;
    snprintf(buf, sizeof(buf), "%s/%s", REDIRECT_DIR, name);
    return buf;
}
int shm_open(const char *name, int oflag, mode_t mode) {
    static int created = 0;
    if (!created) { mkdir(REDIRECT_DIR, 01777); created = 1; }
    int fd = open(redirect_path(name), oflag, mode);
    if (fd >= 0 && (oflag & O_CREAT)) { fchmod(fd, mode); }
    return fd;
}
int shm_unlink(const char *name) { return unlink(redirect_path(name)); }
CEOF
    gcc -shared -fPIC -o /tmp/shm_redirect.so /tmp/shm_redirect.c -ldl
fi
mkdir -p /tmp/shm && chmod 1777 /tmp/shm

export LD_PRELOAD="/tmp/shm_redirect.so${LD_PRELOAD:+:$LD_PRELOAD}"

# Clean stale shared memory from prior crashed runs (both the real /dev/shm and our redirect dir)
rm -f /dev/shm/paddle_* /dev/shm/*signal* /dev/shm/key_caches_* /dev/shm/value_caches_* \
      /dev/shm/__KMP_REGISTERED_LIB_* /dev/shm/sem.mp-* /dev/shm/*.8302 /dev/shm/8300.* \
      /dev/shm/router_* /dev/shm/fmq_* /dev/shm/triton_* 2>/dev/null
rm -f /tmp/shm/paddle_* /tmp/shm/*signal* /tmp/shm/key_caches_* /tmp/shm/value_caches_* \
      /tmp/shm/__KMP_REGISTERED_LIB_* /tmp/shm/sem.mp-* /tmp/shm/*.8302 /tmp/shm/8300.* \
      /tmp/shm/router_* /tmp/shm/fmq_* /tmp/shm/triton_* 2>/dev/null

# Bypass corp HTTP proxy for localhost so benchmark client can reach FastDeploy server directly
export NO_PROXY="127.0.0.1,localhost,0.0.0.0"
export no_proxy="$NO_PROXY"
