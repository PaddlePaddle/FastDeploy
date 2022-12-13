# 1. Install basic software
apt update
apt-get install -y --no-install-recommends \
  gcc g++ git make wget python unzip

# 2. Install arm gcc toolchains
apt-get install -y --no-install-recommends \
  g++-arm-linux-gnueabi gcc-arm-linux-gnueabi \
  g++-arm-linux-gnueabihf gcc-arm-linux-gnueabihf \
  gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# 3. Install cmake 3.10 or above
wget -c https://mms-res.cdn.bcebos.com/cmake-3.10.3-Linux-x86_64.tar.gz && \
  tar xzf cmake-3.10.3-Linux-x86_64.tar.gz && \
  mv cmake-3.10.3-Linux-x86_64 /opt/cmake-3.10 && \
  ln -s /opt/cmake-3.10/bin/cmake /usr/bin/cmake && \
  ln -s /opt/cmake-3.10/bin/ccmake /usr/bin/ccmake
