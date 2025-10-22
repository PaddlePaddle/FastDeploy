/**
 * @file kvcache_connection.h
 * @brief RDMA connection management for key-value cache
 * @version 1.0.0
 * @copyright Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FASTDEPLOY_KVCACHE_CONNECTION_H
#define FASTDEPLOY_KVCACHE_CONNECTION_H

#pragma once

#include <arpa/inet.h>
#include <fcntl.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>
#include <sys/epoll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "kvcache_rdma.h"
#include "util.h"

#define KVCACHE_RDMA_NIC_MAX_LEN 256
#define KVCACHE_RDMA_MAX_NICS 8
#define NAME_MAX 255
#define MAXNAMESIZE 64
#define NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE 16

/// @brief IB device information structure
struct IbDeviceInfo {
  int device;
  uint64_t guid;
  enum ibv_mtu mtu;
  uint64_t busid;
  uint8_t port;
  uint8_t link;
  uint8_t active_mtu;
  int speed;
  ibv_context* context;
  char devName[64];
  int realPort;
  int maxQp;
};

/// @brief Queue Pair information for RDMA
struct QpInfo {
  uint32_t lid;
  uint32_t qpn;
  uint32_t psn;
  union ibv_gid gid;
  enum ibv_mtu mtu;

  /// @brief Serialize QP info to buffer
  void serialize(char* buffer) const {
    uint32_t* intBuffer = reinterpret_cast<uint32_t*>(buffer);
    intBuffer[0] = htonl(lid);
    intBuffer[1] = htonl(qpn);
    intBuffer[2] = htonl(psn);
    memcpy(buffer + 12, gid.raw, sizeof(gid.raw));
    intBuffer[7] = htonl(static_cast<uint32_t>(mtu));
  }

  /// @brief Deserialize QP info from buffer
  void deserialize(const char* buffer) {
    const uint32_t* intBuffer = reinterpret_cast<const uint32_t*>(buffer);
    lid = ntohl(intBuffer[0]);
    qpn = ntohl(intBuffer[1]);
    psn = ntohl(intBuffer[2]);
    memcpy(gid.raw, buffer + 12, sizeof(gid.raw));
    mtu = static_cast<ibv_mtu>(ntohl(intBuffer[7]));
  }

  static const size_t size = 12 + sizeof(gid.raw) + 4;
};

/// @brief RDMA connection context
struct Connection {
  std::atomic<int> connected;

  // Memory regions
  struct ibv_mr* recv_mr;
  struct ibv_mr* send_mr;

  // Cache pointers
  std::vector<std::vector<void*>> local_cache_key_ptr_per_layer;
  std::vector<std::vector<void*>> local_cache_value_ptr_per_layer;

  // Memory region lists
  std::vector<ibv_mr*> write_cache_key_server_mr_list;
  std::vector<ibv_mr*> write_cache_value_server_mr_list;
  std::vector<std::vector<ibv_mr*>> write_mr_key_list;
  std::vector<std::vector<ibv_mr*>> write_mr_value_list;

  // Remote access information
  std::vector<void*> write_cache_key_remote_ptr_list;
  std::vector<uint32_t> write_cache_key_remote_rkey_list;
  std::vector<void*> write_cache_value_remote_ptr_list;
  std::vector<uint32_t> write_cache_value_remote_rkey_list;

  // Received remote memory information
  std::vector<void*> receive_write_cache_key_remote_ptr_list;
  std::vector<uint32_t> receive_write_cache_key_remote_rkey_list;
  std::vector<void*> receive_write_cache_value_remote_ptr_list;
  std::vector<uint32_t> receive_write_cache_value_remote_rkey_list;

  std::vector<void*> send_write_cache_key_remote_ptr_list;
  std::vector<uint32_t> send_write_cache_key_remote_rkey_list;
  std::vector<void*> send_write_cache_value_remote_ptr_list;
  std::vector<uint32_t> send_write_cache_value_remote_rkey_list;

  // For rdma read operations
  std::vector<void*> read_bufs;
  std::vector<ibv_mr*> read_mrs;

  // Work completion tracking
  int wc_count;
  int wc_target_count;

  // Configuration
  int layer_number;
  int block_number;
  int block_byte_size;
  std::string url;

  Connection() = default;
  ~Connection();
};

/// @brief RDMA context structure
struct RdmaContext {
  int sock_fd;
  struct ibv_context* context;
  struct ibv_comp_channel* channel;
  struct ibv_pd* pd;
  struct ibv_mr* mr;
  struct ibv_cq* cq;
  struct ibv_qp* qp;
  struct ibv_port_attr portinfo;
  struct Connection conn;
};

// Global variables
extern std::vector<IbDeviceInfo> g_ib_all_devs;
static int g_kvcache_ib_dev_nums = -1;

// Connection management functions
bool client_exchange_destinations(struct RdmaContext* ctx,
                                  int ib_port,
                                  unsigned int port,
                                  int gidx,
                                  const std::string& dst_ip);

int server_exchange_qp_info(int connfd, QpInfo* local_dest, QpInfo* rem_dest);
struct RdmaContext* create_qp(struct IbDeviceInfo* ib_dev,
                              struct ibv_pd** g_pd);
bool clear_qp_info(struct RdmaContext* ctx);

// QP modification functions
QpStatus modify_qp_to_rts(struct RdmaContext* ctx,
                          int port,
                          int my_psn,
                          struct QpInfo* dest,
                          int sgid_id);
bool poll_cq_with_timeout(struct RdmaContext* ctx,
                          int timeout_seconds,
                          int cqe_count);

// Utility functions
int get_port_info(struct ibv_context* Context,
                  int port,
                  struct ibv_port_attr* attr);
int parse_port_ib_info();

// Memory region exchange
bool client_exchange_mr(struct RdmaContext* ctx);
bool server_exchange_mr(struct RdmaContext* ctx);
bool server_send_memory_region(struct RdmaContext* ctx,
                               void* local_mr,
                               int byte_num);
bool client_receive_memory_region(struct RdmaContext* ctx,
                                  void* remote_mr,
                                  int byte_num);

// Network setup
int setup_listening_socket(int port);
int configure_epoll(int sockfd);
std::vector<std::string> get_net_ifname();

#endif  // FASTDEPLOY_KVCACHE_CONNECTION_H
