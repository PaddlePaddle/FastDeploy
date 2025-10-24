/**
 * @file kvcache_rdma.cpp
 * @brief RDMA-based Key-Value Cache Communication Implementation
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
#include "kvcache_rdma.h"
#include "kvcache_connection.h"
#include "log.h"
#include "util.h"

#include <fcntl.h>
#include <netdb.h>
#include <rdma/rdma_cma.h>
#include <sys/epoll.h>
#include <unistd.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <stdexcept>
#include <thread>

/**
 * @brief Construct a new RDMACommunicator object
 *
 * @param role Role in distributed system ("decode" or "prefill")
 * @param gpu_idx GPU device index to use
 * @param port Communication port number
 * @param local_key_cache Vector of local key cache pointers
 * @param local_value_cache Vector of local value cache pointers
 * @param block_number Number of blocks in cache
 * @param block_bytes Size of each block in bytes
 *
 * @throws std::runtime_error If initialization fails
 */
RDMACommunicator::RDMACommunicator(std::string& role,
                                   int gpu_idx,
                                   std::string& port,
                                   std::vector<int64_t> local_key_cache,
                                   std::vector<int64_t> local_value_cache,
                                   int block_number,
                                   int block_bytes)
    : splitwise_role(role),
      gpu_idx(gpu_idx),
      port(port),
      local_cache_key_ptr_layer_head_(std::move(local_key_cache)),
      local_cache_value_ptr_layer_head_(std::move(local_value_cache)),
      block_number(block_number),
      block_size_byte(block_bytes),
      RDMACommunicator_status(0),
      rdma_event_channel_epoll_fd(-1) {
  try {
    WARN("Initializing RDMA communicator for role: %s", role.c_str());

    // Step 1: Initialize KV cache config
    KVCacheConfig::getInstance().displayConfiguration();

    // Step 2: Initialize KV cache structure
    // Validate and set number of layers
    layer_number = static_cast<int>(local_cache_key_ptr_layer_head_.size());
    if (layer_number <= 0) {
      throw std::runtime_error("Invalid layer number");
    }

    // Step 2: Setup cache vectors and pointers
    resize_vectors();
    assign_pointers();

    // Step 3:Initialize the event channel
    rdma_event_channel_epoll_fd = epoll_create1(EPOLL_CLOEXEC);
    if (rdma_event_channel_epoll_fd < 0) {
      throw std::runtime_error("Failed to create epoll fd: " +
                               std::string(strerror(errno)));
    }

    // Start the server thread (if in decode role)
    if (splitwise_role == "decode") {
      std::thread server_thread([this]() {
        try {
          this->init_server();
        } catch (const std::exception& e) {
          ERR("Server thread failed: %s", e.what());
        }
      });
      server_thread.detach();
    }

    RDMACommunicator_status = 1;
    INFO("RDMA communicator initialized successfully");
  } catch (const std::exception& e) {
    ERR("Initialization failed: %s", e.what());
    if (rdma_event_channel_epoll_fd >= 0) {
      close(rdma_event_channel_epoll_fd);
      rdma_event_channel_epoll_fd = -1;
    }
    throw;
  }
}

void RDMACommunicator::resize_vectors() {
  if (layer_number <= 0) {
    throw std::runtime_error("Invalid layer number");
  }

  local_cache_key_ptr_per_layer.resize(layer_number);
  local_cache_value_ptr_per_layer.resize(layer_number);
}

void RDMACommunicator::assign_pointers() {
  // Validate block configuration
  if (block_number <= 0 || block_size_byte <= 0) {
    throw std::runtime_error("Invalid block configuration");
  }

  // Assign pointers for each layer and block
  for (int layer_idx = 0; layer_idx < layer_number; ++layer_idx) {
    // Validate layer head pointers
    if (local_cache_key_ptr_layer_head_[layer_idx] == 0 ||
        local_cache_value_ptr_layer_head_[layer_idx] == 0) {
      throw std::runtime_error("Invalid cache pointer for layer " +
                               std::to_string(layer_idx));
    }

    // Resize block vectors for current layer
    local_cache_key_ptr_per_layer[layer_idx].resize(block_number);
    local_cache_value_ptr_per_layer[layer_idx].resize(block_number);

    // Calculate and assign block pointers
    for (int block_idx = 0; block_idx < block_number; ++block_idx) {
      local_cache_key_ptr_per_layer[layer_idx][block_idx] =
          reinterpret_cast<void*>(local_cache_key_ptr_layer_head_[layer_idx] +
                                  block_idx * block_size_byte);

      local_cache_value_ptr_per_layer[layer_idx][block_idx] =
          reinterpret_cast<void*>(local_cache_value_ptr_layer_head_[layer_idx] +
                                  block_idx * block_size_byte);
    }
  }
}

void RDMACommunicator::validate_addr() {
  if (main_ip_list.empty()) {
    throw std::runtime_error("main_ip_list is empty");
  } else {
    if (!main_ip_list.empty()) {
      LOGD("Local main NIC addresses:");
      for (const auto& nic_ip : main_ip_list) {
        LOGD("- %s", nic_ip.c_str());
      }
    }
  }
}

RDMACommunicator::~RDMACommunicator() {
  try {
    WARN("Destroying RDMA communicator");

    // Mark as closed/shutdown state
    RDMACommunicator_status = 0;

    // Clean up all connections
    {
      std::lock_guard<std::mutex> lock(mutex_);
      conn_map.clear();
    }

    // Clean up memory regions
    auto deregister_mrs = [](std::vector<ibv_mr*>& mrs, const char* name) {
      for (auto* mr : mrs) {
        if (mr && ibv_dereg_mr(mr)) {
          ERR("Failed to deregister %s MR: %s", name, strerror(errno));
        }
      }
      mrs.clear();
    };

    deregister_mrs(write_mr_key_list, "write key");
    deregister_mrs(write_mr_value_list, "write value");
    deregister_mrs(write_cache_key_server_mr_list, "server key");
    deregister_mrs(write_cache_value_server_mr_list, "server value");

    // Clean up protection domain
    if (g_pd) {
      if (ibv_dealloc_pd(g_pd)) {
        ERR("Failed to deallocate protection domain: %s", strerror(errno));
      }
      g_pd = nullptr;
    }

    // Close event channel
    if (rdma_event_channel_epoll_fd >= 0) {
      close(rdma_event_channel_epoll_fd);
      rdma_event_channel_epoll_fd = -1;
    }

    WARN("RDMA communicator destroyed successfully");
  } catch (const std::exception& e) {
    ERR("Destruction failed: %s", e.what());
  }
}

int RDMACommunicator::start_server(int sport, int sgid_idx, int gpu_index) {
  WARN("verbs server starting …");

  int sockfd = setup_listening_socket(sport);
  if (sockfd < 0) {
    ERR("Failed to set up listening socket");
    return -1;
  }

  if (g_ib_all_devs.size() == 0) {
    if (parse_port_ib_info() != 0) {
      ERR("decode parse_port_ib_info error, please set rdma nics info");
      return -1;
    }
  }

  int use_event = 1;
  int epollfd = configure_epoll(sockfd);
  if (epollfd < 0) {
    ERR("Failed to configure epoll");
    close(sockfd);
    return -1;
  }

  struct epoll_event ev, events[10];
  char buffer[QpInfo::size] = {0};
  std::map<int, struct RdmaContext*> connectionContexts;
  std::unique_ptr<QpInfo> rem_dest(new QpInfo());
  std::unique_ptr<QpInfo> local_dest(new QpInfo());
  struct RdmaContext* contexts[RDMA_TCP_CONNECT_SIZE] = {nullptr};

  while (RDMACommunicator_status == 1) {
    int nfds = epoll_wait(epollfd, events, 10, -1);
    if (nfds < 0) {
      if (errno == EINTR) continue;
      ERR("epoll_wait failed: %s", strerror(errno));
      break;
    }

    for (int i = 0; i < nfds; i++) {
      int event_fd = events[i].data.fd;

      if (event_fd == sockfd) {
        int connfd = accept(sockfd, nullptr, nullptr);
        if (connfd < 0) {
          if (errno == EINTR) continue;
          ERR("accept() failed: %s", strerror(errno));
          continue;
        }

        if (fcntl(connfd, F_SETFL, fcntl(connfd, F_GETFL, 0) | O_NONBLOCK) <
            0) {
          ERR("Failed to set non-blocking mode for connfd: %s",
              strerror(errno));
          close(connfd);
          continue;
        }

        ev.events = EPOLLIN | EPOLLRDHUP | EPOLLERR;
        ev.data.fd = connfd;
        if (epoll_ctl(epollfd, EPOLL_CTL_ADD, connfd, &ev) < 0) {
          ERR("Failed to add connfd to epoll: %s", strerror(errno));
          close(connfd);
          continue;
        }

        size_t dev_idx = gpu_index % g_ib_all_devs.size();
        struct IbDeviceInfo* ib_dev = &g_ib_all_devs[dev_idx];
        struct RdmaContext* ctx = create_qp(ib_dev, &g_pd);
        if (!ctx) {
          ERR("Failed to initialize RDMA Context");
          close_server_connection(connfd, ctx, epollfd, connectionContexts);
          continue;
        }

        connectionContexts[connfd] = ctx;
        ctx->conn.layer_number = layer_number;
        ctx->conn.block_number = block_number;
        ctx->conn.block_byte_size = block_size_byte;
        ctx->conn.local_cache_key_ptr_per_layer = local_cache_key_ptr_per_layer;
        ctx->conn.local_cache_value_ptr_per_layer =
            local_cache_value_ptr_per_layer;

        std::lock_guard<std::mutex> lock(mutex_);
        if (!server_mr_register_per_layer(ctx)) {
          ERR("server_mr_register_per_layer failed");
          return -1;
        }

        if (get_port_info(ctx->context, ib_dev->port, &ctx->portinfo)) {
          close_server_connection(connfd, ctx, epollfd, connectionContexts);
          ERR("Couldn't get port info");
          continue;
        }

        local_dest->lid = ctx->portinfo.lid;
        local_dest->mtu = ctx->portinfo.active_mtu;
        if (ctx->portinfo.link_layer != IBV_LINK_LAYER_ETHERNET &&
            !local_dest->lid) {
          close_server_connection(connfd, ctx, epollfd, connectionContexts);
          ERR("Couldn't get local LID");
          continue;
        }

        if (sgid_idx >= 0) {
          if (ibv_query_gid(
                  ctx->context, ib_dev->port, sgid_idx, &local_dest->gid)) {
            close_server_connection(connfd, ctx, epollfd, connectionContexts);
            ERR("Can't read sgid of index %d", sgid_idx);
            continue;
          }
        } else {
          memset(&local_dest->gid, 0, sizeof local_dest->gid);
        }

        local_dest->qpn = ctx->qp->qp_num;

        if (server_exchange_qp_info(connfd, local_dest.get(), rem_dest.get()) <
            0) {
          close_server_connection(connfd, ctx, epollfd, connectionContexts);
          ERR("Failed to exchange QP info");
          continue;
        }

        if (modify_qp_to_rts(
                ctx, ib_dev->port, local_dest->psn, rem_dest.get(), sgid_idx) !=
            QpStatus::kSuccess) {
          close_server_connection(connfd, ctx, epollfd, connectionContexts);
          ERR("Failed to connect to remote QP");
          continue;
        }

        server_exchange_mr(ctx);
      } else {
        auto ctx_iter = connectionContexts.find(event_fd);
        if (ctx_iter == connectionContexts.end()) {
          LOGD("Unknown Connection fd: %d", event_fd);
          continue;
        }
        struct RdmaContext* ctx = ctx_iter->second;
        if (events[i].events & (EPOLLRDHUP | EPOLLHUP | EPOLLERR)) {
          LOGD("Connection closed or error detected on fd: %d", event_fd);
          close_server_connection(event_fd, ctx, epollfd, connectionContexts);
          continue;
        }

        if (events[i].events & EPOLLIN) {
          char buffer[sizeof(QpInfo)];
          ssize_t bytes_read = read(event_fd, buffer, sizeof(buffer));

          if (bytes_read <= 0) {
            LOGD("Read error or peer closed Connection on fd %d", event_fd);
            close_server_connection(event_fd, ctx, epollfd, connectionContexts);
          }
        }
      }
    }
  }

  close(sockfd);
  close(epollfd);
  return 0;
}

void RDMACommunicator::close_server_connection(
    int fd,
    struct RdmaContext* ctx,
    int epollfd,
    std::map<int, struct RdmaContext*>& connectionContexts) {
  if (ctx) {
    if (!deregister_memory_regions(ctx)) {
      WARN("Failed to clear memory regions for Connection fd %d", fd);
    }
    if (!clear_qp_info(ctx)) {
      WARN("Failed to clear memory regions for Connection fd %d", fd);
    }
    delete ctx;
  }
  connectionContexts.erase(fd);
  epoll_ctl(epollfd, EPOLL_CTL_DEL, fd, nullptr);
  close(fd);
  LOGD("Connection fd %d closed and cleaned up", fd);
}

void RDMACommunicator::close_client_connection(int fd,
                                               struct RdmaContext* ctx,
                                               int epollfd) {
  if (!ctx) {
    LOGD("ctx is NULL, skipping cleanup for fd %d", fd);
    epoll_ctl(epollfd, EPOLL_CTL_DEL, fd, nullptr);
    close(fd);
    return;
  }

  conn_map.erase(ctx->conn.url);

  for (size_t i = 0; i < ctx->conn.read_bufs.size(); ++i) {
    if (ctx->conn.read_mrs[i]) ibv_dereg_mr(ctx->conn.read_mrs[i]);
    if (ctx->conn.read_bufs[i]) free(ctx->conn.read_bufs[i]);
  }
  ctx->conn.read_bufs.clear();
  ctx->conn.read_mrs.clear();

  ctx->conn.connected = 0;
  if (!clear_qp_info(ctx)) {
    LOGD("Failed to clear memory regions for Connection fd %d", fd);
  }

  epoll_ctl(epollfd, EPOLL_CTL_DEL, fd, nullptr);
  close(fd);
  delete ctx;
  LOGD("Connection fd %d closed and cleaned up", fd);
}

bool RDMACommunicator::deregister_memory_regions(struct RdmaContext* ctx) {
  if (ctx == nullptr) {
    ERR("Context is null, cannot clear server Connection.");
    return false;
  }

  for (int layer_idx = 0; layer_idx < layer_number; layer_idx++) {
    if (!write_mr_key_list.empty() && !write_mr_value_list.empty()) {
      if (ibv_dereg_mr(write_mr_key_list[layer_idx])) {
        ERR("Failed to deregister memory region: write_mr_key_list, layer %d",
            layer_idx);
      }
      if (ibv_dereg_mr(write_mr_value_list[layer_idx])) {
        ERR("Failed to deregister memory region: write_mr_value_list, layer %d",
            layer_idx);
      }
    }
  }
  return true;
}

/**
 * Initialize the RDMA server
 *
 * @return Result code: 0 on success, negative value on failure
 */
int RDMACommunicator::init_server() {
  WARN("Initializing RDMA server...");
  return start_server(KVCacheConfig::getInstance().resolve_rdma_dest_port(port),
                      KVCacheConfig::getInstance().get_rdma_gid_index(),
                      gpu_idx);
}

/**
 * Fetch the local IP address from the main IP list
 *
 * @return The first IP address in the main IP list, or empty string if list is
 * empty
 */
std::string RDMACommunicator::fetch_local_ip() {
  if (main_ip_list.empty()) {
    ERR("Error: main_ip_list are empty.");
    return nullptr;
  }

  return main_ip_list[0];
}

/**
 * Connect to a remote RDMA endpoint
 *
 * Establishes an RDMA connection with the specified destination IP and port.
 *
 * @param dst_ip Destination IP address
 * @param dst_port Destination port
 * @return ConnStatus::kConnected ConnStatus::kError;
 */

int RDMACommunicator::connect(const std::string& dst_ip,
                              const std::string& dst_port) {
  std::string url = dst_ip + ":" + dst_port;

  // Initialize IB devices if not already done
  if (g_ib_all_devs.size() == 0) {
    if (parse_port_ib_info() != 0) {
      ERR("prefill parse_port_ib_info is error, please set rdma nics info");
      return static_cast<int>(ConnStatus::kInvalidParameters);
    }
  }

  // Check if already connected
  if (is_connected(dst_ip, dst_port)) {
    INFO("Already connected to %s:%s", dst_ip.c_str(), dst_port.c_str());
    return static_cast<int>(ConnStatus::kConnected);
  }

  // Create Queue Pair (QP) for the connection
  size_t dev_idx = gpu_idx % g_ib_all_devs.size();
  struct IbDeviceInfo* ib_dev = &g_ib_all_devs[dev_idx];
  struct RdmaContext* ctx = create_qp(ib_dev, &g_pd);
  if (!ctx) {
    ERR("Couldn't create QP");
    return static_cast<int>(ConnStatus::kError);
  }

  // Initialize connection data
  ctx->conn.url = url;
  ctx->conn.layer_number = layer_number;
  ctx->conn.block_number = block_number;
  ctx->conn.block_byte_size = block_size_byte;

  // Get port information for the connection
  if (get_port_info(ctx->context, ib_dev->port, &ctx->portinfo)) {
    ERR("Couldn't get port info");
    return static_cast<int>(ConnStatus::kError);
  }
  // Register memory regions
  if (!client_mr_register_per_layer(ctx)) {
    ERR("server_mr_register_per_layer failed");
    return static_cast<int>(ConnStatus::kError);
  }

  // Exchange connection information with remote peer
  if (!client_exchange_destinations(
          ctx,
          ib_dev->port,
          KVCacheConfig::getInstance().resolve_rdma_dest_port(dst_port),
          KVCacheConfig::getInstance().get_rdma_gid_index(),
          dst_ip)) {
    ERR("Couldn't getexchange port infodestinations");
    return static_cast<int>(ConnStatus::kError);
  } else {
    std::lock_guard<std::mutex> lock(mutex_);
    ctx->conn.connected = 1;
    conn_map[url] = ctx;
    client_exchange_mr(ctx);
  }

  // Allocate RDMA read and register read buffers
  ctx->conn.read_bufs.resize(block_number, nullptr);
  ctx->conn.read_mrs.resize(block_number, nullptr);

  for (size_t i = 0; i < block_number; ++i) {
    // Allocate memory for read buffer
    ctx->conn.read_bufs[i] = malloc(block_size_byte);
    if (!ctx->conn.read_bufs[i]) {
      ERR("Failed to allocate read buffer");
      return static_cast<int>(ConnStatus::kError);
    }
    // Register memory region for read buffer
    ctx->conn.read_mrs[i] = ibv_reg_mr(ctx->pd,
                                       ctx->conn.read_bufs[i],
                                       block_size_byte,
                                       IBV_ACCESS_LOCAL_WRITE);
    if (!ctx->conn.read_mrs[i]) {
      ERR("Failed to register memory for RDMA Read buffer");
      return static_cast<int>(ConnStatus::kError);
    }
  }

  // Start client listener thread if not already started
  if (start_client_listener == false) {
    std::thread client_thread =
        std::thread([this]() { this->client_listener(); });
    if (client_thread.joinable()) {
      client_thread.detach();
      std::lock_guard<std::mutex> lock(mutex_);
    }
    start_client_listener = true;
  }

  // Add socket to epoll for event monitoring
  if (ctx->sock_fd != 0) {
    struct epoll_event ev;
    ev.events = EPOLLIN | EPOLLOUT | EPOLLERR;
    ev.data.ptr = ctx;
    int ret = epoll_ctl(
        rdma_event_channel_epoll_fd, EPOLL_CTL_ADD, ctx->sock_fd, &ev);
    if (ret != 0) {
      ERR("failed to add event channel %d", ret);
      return static_cast<int>(ConnStatus::kError);
    }
  }

  WARN("connect end ....");
  return static_cast<int>(ConnStatus::kConnected);
}

int RDMACommunicator::client_listener() {
  struct epoll_event events[10];

  while (RDMACommunicator_status == 1) {
    int nfds = epoll_wait(rdma_event_channel_epoll_fd, events, 10, -1);
    if (nfds < 0) {
      if (errno == EINTR) {
        WARN("epoll_wait interrupted, continuing...");
        continue;
      }
      ERR("epoll_wait failed: %s", strerror(errno));
      return -1;
    }

    for (int i = 0; i < nfds; ++i) {
      RdmaContext* ctx = static_cast<RdmaContext*>(events[i].data.ptr);
      if (!ctx) {
        ERR("Null context received in epoll event");
        continue;
      }

      if (events[i].events & (EPOLLERR | EPOLLHUP | EPOLLRDHUP)) {
        int err = 0;
        socklen_t len = sizeof(err);
        getsockopt(ctx->sock_fd, SOL_SOCKET, SO_ERROR, &err, &len);
        if (err) ERR("Socket error: %s", strerror(err));

        std::lock_guard<std::mutex> lock(mutex_);
        close_client_connection(ctx->sock_fd, ctx, rdma_event_channel_epoll_fd);
        continue;
      }

      if (events[i].events & EPOLLIN) {
        char buffer[sizeof(QpInfo)];
        ssize_t bytes_read = read(ctx->sock_fd, buffer, sizeof(buffer));

        if (bytes_read <= 0) {
          if (bytes_read == 0) {
            WARN("Peer closed connection on fd %d", ctx->sock_fd);
          } else {
            ERR("Read error on fd %d: %s", ctx->sock_fd, strerror(errno));
          }

          std::lock_guard<std::mutex> lock(mutex_);
          close_client_connection(
              ctx->sock_fd, ctx, rdma_event_channel_epoll_fd);
        }
      }
    }
  }

  return 0;
}

bool RDMACommunicator::is_connected(const std::string& dst_ip,
                                    const std::string& dst_port) {
  std::string url = dst_ip + ":" + dst_port;
  return conn_map.find(url) != conn_map.end();
}

void RDMACommunicator::remove_conn(const std::string& url) {
  if (conn_map.find(url) != conn_map.end()) {
    struct RdmaContext* ctx = conn_map[url];
    conn_map.erase(url);
    free(ctx->context);
  }
}

struct RdmaContext* RDMACommunicator::get_conn(const std::string& ip,
                                               const std::string& port) {
  std::string url = ip + ":" + port;
  if (conn_map.find(url) == conn_map.end()) {
    return NULL;
  }
  return conn_map[url];
}

/**
 * @brief Register a memory region
 * @param pd Pointer to the protection domain
 * @param addr Starting address of the memory
 * @param size Size of the memory
 * @param desc Description of the memory region
 * @param access_flags Access flags
 * @return Pointer to the registered memory region on success
 * @throws std::runtime_error Throws an exception if registration fails
 */
struct ibv_mr* RDMACommunicator::register_memory_region(ibv_pd* pd,
                                                        void* addr,
                                                        size_t size,
                                                        const std::string& desc,
                                                        uint32_t access_flags) {
  if (!pd || !addr || size == 0) {
    throw std::invalid_argument("Invalid memory region parameters");
  }

  // Check and set the Relaxed Ordering flag
  if (KVCacheConfig::getInstance().is_relax_ordering_enabled()) {
    access_flags |= IBV_ACCESS_RELAXED_ORDERING;
    LOGD("Enabled Relaxed Ordering for %s", desc.c_str());
  }

  struct ibv_mr* mr = ibv_reg_mr(pd, addr, size, access_flags);
  if (!mr) {
    throw std::runtime_error("Failed to register memory region " + desc + ": " +
                             strerror(errno));
  }

  LOGD("Registered %s MR: addr=%p, size=%zu, flags=0x%x, lkey=0x%x",
       desc.c_str(),
       addr,
       size,
       access_flags,
       mr->lkey);
  return mr;
}

/**
 * @brief Register client memory regions
 * @param ctx Pointer to the RDMA context
 * @note This method registers memory regions for the KV cache of each layer
 */
bool RDMACommunicator::client_mr_register_per_layer(RdmaContext* ctx) {
  if (!ctx || !ctx->pd) {
    ERR("Invalid RDMA context");
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  if (!write_mr_key_list.empty() || !write_mr_value_list.empty()) {
    WARN("Memory regions already registered");
    return true;
  }

  const size_t list_size = layer_number;
  write_mr_key_list.resize(list_size, nullptr);
  write_mr_value_list.resize(list_size, nullptr);

  const uint32_t access_flags =
      IBV_ACCESS_LOCAL_WRITE |
      (KVCacheConfig::getInstance().is_relax_ordering_enabled()
           ? IBV_ACCESS_RELAXED_ORDERING
           : 0);

  for (int i = 0; i < static_cast<int>(list_size); ++i) {
    void* key_ptr = reinterpret_cast<void*>(local_cache_key_ptr_layer_head_[i]);
    void* val_ptr =
        reinterpret_cast<void*>(local_cache_value_ptr_layer_head_[i]);
    size_t size = static_cast<size_t>(block_size_byte) * block_number;

    write_mr_key_list[i] =
        register_memory_region(ctx->pd,
                               key_ptr,
                               size,
                               "client_key_" + std::to_string(i),
                               access_flags);
    if (!write_mr_key_list[i]) goto fail;

    write_mr_value_list[i] =
        register_memory_region(ctx->pd,
                               val_ptr,
                               size,
                               "client_value_" + std::to_string(i),
                               access_flags);
    if (!write_mr_value_list[i]) goto fail;
  }

  return true;

fail:
  ERR("Memory region registration failed. Cleaning up...");

  for (auto* mr : write_mr_key_list) {
    if (mr) ibv_dereg_mr(mr);
  }
  for (auto* mr : write_mr_value_list) {
    if (mr) ibv_dereg_mr(mr);
  }

  write_mr_key_list.clear();
  write_mr_value_list.clear();
  return false;
}

/**
 * @brief Register server-side memory regions for RDMA operations
 * @param ctx RDMA context containing protection domain and other resources
 *
 * @details This method registers memory regions for both keys and values
 *          for each layer, enabling remote read/write access.
 */
bool RDMACommunicator::server_mr_register_per_layer(RdmaContext* ctx) {
  if (!ctx || !ctx->pd) {
    ERR("Invalid RDMA context");
    return false;
  }

  write_cache_key_server_mr_list.clear();
  write_cache_value_server_mr_list.clear();

  const uint32_t access_flags =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

  for (int i = 0; i < layer_number; ++i) {
    void* key_ptr = reinterpret_cast<void*>(local_cache_key_ptr_layer_head_[i]);
    void* val_ptr =
        reinterpret_cast<void*>(local_cache_value_ptr_layer_head_[i]);
    size_t size = static_cast<size_t>(block_size_byte) * block_number;

    struct ibv_mr* key_mr = register_memory_region(
        ctx->pd, key_ptr, size, "key_" + std::to_string(i), access_flags);
    if (!key_mr) {
      ERR("Failed to register key MR at layer %d", i);
      goto fail;
    }

    struct ibv_mr* value_mr = register_memory_region(
        ctx->pd, val_ptr, size, "value_" + std::to_string(i), access_flags);
    if (!value_mr) {
      ERR("Failed to register value MR at layer %d", i);
      ibv_dereg_mr(key_mr);
      goto fail;
    }

    write_cache_key_server_mr_list.push_back(key_mr);
    write_cache_value_server_mr_list.push_back(value_mr);
  }

  ctx->conn.write_cache_key_server_mr_list = write_cache_key_server_mr_list;
  ctx->conn.write_cache_value_server_mr_list = write_cache_value_server_mr_list;
  return true;

fail:
  for (auto* mr : write_cache_key_server_mr_list) {
    if (mr) ibv_dereg_mr(mr);
  }
  for (auto* mr : write_cache_value_server_mr_list) {
    if (mr) ibv_dereg_mr(mr);
  }

  write_cache_key_server_mr_list.clear();
  write_cache_value_server_mr_list.clear();
  return false;
}

int RDMACommunicator::write_cache(const std::string& ip,
                                  const std::string& port,
                                  const std::vector<int64_t>& local_block_ids,
                                  const std::vector<int64_t>& remote_block_ids,
                                  int32_t layer_idx) {
  // Parameter validation
  if (local_block_ids.size() != remote_block_ids.size()) {
    ERR("Block ID lists size mismatch: local=%zu, remote=%zu",
        local_block_ids.size(),
        remote_block_ids.size());
    return -1;
  }

  if (layer_idx < 0 || layer_idx >= layer_number) {
    ERR("Invalid layer index: %d (max: %d)", layer_idx, layer_number - 1);
    return -1;
  }

  const auto block_num = local_block_ids.size();
  if (block_num == 0) {
    WARN("Empty block list, nothing to write");
    return 0;
  }

  // Performance debugging
  std::chrono::steady_clock::time_point start_time;
  if (KVCacheConfig::getInstance().is_debug_mode_enabled()) {
    start_time = std::chrono::steady_clock::now();
  }

  // Get connection context with thread safety
  std::unique_lock<std::mutex> lock(mutex_);
  auto* ctx = get_conn(ip, port);
  lock.unlock();

  if (!ctx || !ctx->conn.connected) {
    ERR("No active connection to %s:%s", ip.c_str(), port.c_str());
    return -1;
  }

  std::vector<uint64_t> cache_key_remote_addr(block_num);
  std::vector<uint64_t> cache_value_remote_addr(block_num);
  std::vector<uint64_t> crc_cache_key_remote_addr(block_num);
  std::vector<uint64_t> crc_cache_value_remote_addr(block_num);

  uint32_t cache_key_rkey =
      ctx->conn.write_cache_key_remote_rkey_list[layer_idx];
  uint32_t cache_value_rkey =
      ctx->conn.write_cache_value_remote_rkey_list[layer_idx];
  uint32_t crc_cache_key_rkey, crc_cache_value_rkey;

  for (size_t block_index = 0; block_index < block_num; ++block_index) {
    char* char_ptr = static_cast<char*>(
        ctx->conn.write_cache_key_remote_ptr_list[layer_idx]);
    cache_key_remote_addr[block_index] =
        (uint64_t(char_ptr + remote_block_ids[block_index] * block_size_byte));
    char_ptr = static_cast<char*>(
        ctx->conn.write_cache_value_remote_ptr_list[layer_idx]);
    cache_value_remote_addr[block_index] =
        (uint64_t(char_ptr + remote_block_ids[block_index] * block_size_byte));
  }
  ctx->conn.wc_target_count = 0;
  for (int i = 0; i < 2; ++i) {
    bool is_key = (i == 0);
    uint32_t rkey = (is_key ? cache_key_rkey : cache_value_rkey);
    std::vector<uint64_t>& remote_addr =
        (is_key ? cache_key_remote_addr : cache_value_remote_addr);
    if (!post_block_send(ctx,
                         layer_idx,
                         local_block_ids,
                         is_key,
                         remote_addr,
                         rkey,
                         ip,
                         port)) {
      return -1;
    }
  }

  if (KVCacheConfig::getInstance().is_debug_mode_enabled()) {
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now() - start_time)
                           .count();

    DEBUG(
        "Write cache completed - IP: %s, Port: %s, Layer: %d, BlockSize: %d, "
        "Blocks: %lu, Duration: %ld us",
        ip.c_str(),
        port.c_str(),
        layer_idx,
        block_size_byte,
        block_num,
        duration_us);
  }
  return 0;
}

bool RDMACommunicator::post_block_send(
    struct RdmaContext* ctx,
    int layer_idx,
    const std::vector<int64_t>& local_block_ids,
    bool is_key,
    std::vector<uint64_t>& remote_addr,
    uint32_t rkey,
    const std::string& ip,
    const std::string& port) {
  auto block_num = local_block_ids.size();
  assert(block_num > 0 && "block_num must be > 0");

  bool success = execute_rdma_writes(
      ctx, layer_idx, local_block_ids, is_key, remote_addr, rkey);

  if (success) {
    if (KVCacheConfig::getInstance().is_gdrcopy_flush_enabled()) {
      const size_t last_idx = block_num - 1;
      success = execute_read_verification(
          ctx, last_idx, remote_addr[last_idx], rkey, layer_idx, ip, port);
    }
  }

  return success;
}

bool RDMACommunicator::execute_rdma_writes(
    struct RdmaContext* ctx,
    int layer_idx,
    const std::vector<int64_t>& local_block_ids,
    bool is_key,
    std::vector<uint64_t>& remote_addr,
    uint32_t rkey) {
  auto block_num = local_block_ids.size();
  struct ibv_sge* sge_list = new ibv_sge[block_num];
  struct ibv_send_wr* send_wr_list = new ibv_send_wr[block_num];

  prepare_write_requests(sge_list,
                         send_wr_list,
                         layer_idx,
                         local_block_ids,
                         is_key,
                         remote_addr,
                         rkey);

  bool success = true;
  size_t inflight_wr = 0;

  for (size_t scnt = 0; scnt < block_num; ++scnt) {
    size_t idx = scnt % RDMA_WR_LIST_MAX_SIZE;
    inflight_wr++;

    bool is_batch_end =
        (idx == RDMA_WR_LIST_MAX_SIZE - 1 || scnt == block_num - 1);
    bool need_poll = (inflight_wr >= RDMA_SQ_MAX_SIZE || scnt == block_num - 1);

    if (is_batch_end) {
      if (!post_send_with_retry(
              ctx, &send_wr_list[scnt - idx], inflight_wr, need_poll)) {
        success = false;
        break;
      }
      if (need_poll) {
        inflight_wr = 0;
      }
    }
  }

  delete[] sge_list;
  delete[] send_wr_list;
  return success;
}

void RDMACommunicator::prepare_write_requests(
    struct ibv_sge* sge_list,
    struct ibv_send_wr* send_wr_list,
    int layer_idx,
    const std::vector<int64_t>& local_block_ids,
    bool is_key,
    std::vector<uint64_t>& remote_addr,
    uint32_t rkey) {
  auto block_num = local_block_ids.size();

  for (size_t i = 0; i < block_num; ++i) {
    sge_list[i].addr =
        (uintptr_t)(is_key
                        ? local_cache_key_ptr_per_layer[layer_idx]
                                                       [local_block_ids[i]]
                        : local_cache_value_ptr_per_layer[layer_idx]
                                                         [local_block_ids[i]]);
    sge_list[i].length = block_size_byte;
    sge_list[i].lkey = (is_key ? write_mr_key_list[layer_idx]->lkey
                               : write_mr_value_list[layer_idx]->lkey);

    size_t idx = i % RDMA_WR_LIST_MAX_SIZE;
    send_wr_list[i].wr_id = i;
    send_wr_list[i].next =
        (idx == RDMA_WR_LIST_MAX_SIZE - 1 || i == block_num - 1)
            ? nullptr
            : &send_wr_list[i + 1];
    send_wr_list[i].sg_list = &sge_list[i];
    send_wr_list[i].num_sge = 1;
    send_wr_list[i].opcode = IBV_WR_RDMA_WRITE;
    send_wr_list[i].send_flags = (i == block_num - 1) ? IBV_SEND_SIGNALED : 0;
    send_wr_list[i].wr.rdma.remote_addr = remote_addr[i];
    send_wr_list[i].wr.rdma.rkey = rkey;
  }
}

bool RDMACommunicator::post_send_with_retry(struct RdmaContext* ctx,
                                            struct ibv_send_wr* wr_list,
                                            size_t inflight_wr,
                                            bool need_poll) {
  const int max_retries = 7;
  int retries = 0;
  int ret = 0;
  struct ibv_send_wr* bad_wr = nullptr;

  if (inflight_wr >= RDMA_SQ_MAX_SIZE && wr_list) {
    struct ibv_send_wr* last_wr = wr_list;
    while (last_wr->next) {
      last_wr = last_wr->next;
    }
    last_wr->send_flags |= IBV_SEND_SIGNALED;
  }

  do {
    ret = ibv_post_send(ctx->qp, wr_list, &bad_wr);
    if (ret == 0) {
      if (need_poll) {
        ctx->conn.wc_count = 0;
        ctx->conn.wc_target_count = 0;
        if (!poll_cq_with_timeout(ctx, RDMA_POLL_CQE_TIMEOUT, 1)) {
          ERR("Polling CQ failed after RDMA Write");
          return false;
        }
      }
      return true;
    } else {
      ERR("ibv_post_send failed: %s (errno: %d), retry %d/%d",
          strerror(errno),
          errno,
          retries + 1,
          max_retries);
      usleep(1000);
      retries++;
    }
  } while (retries < max_retries);

  ERR("ibv_post_send failed after %d retries: %s (errno: %d)",
      retries,
      strerror(errno),
      errno);
  return false;
}

bool RDMACommunicator::execute_read_verification(struct RdmaContext* ctx,
                                                 size_t block_idx,
                                                 uint64_t remote_addr,
                                                 uint32_t rkey,
                                                 int layer_idx,
                                                 const std::string& ip,
                                                 const std::string& port) {
  ibv_sge read_sge = {
      .addr = reinterpret_cast<uintptr_t>(ctx->conn.read_bufs[block_idx]),
      .length = static_cast<uint32_t>(block_size_byte),
      .lkey = ctx->conn.read_mrs[block_idx]->lkey};

  ibv_send_wr read_wr = {};
  read_wr.wr_id = 1000 + block_idx;
  read_wr.sg_list = &read_sge;
  read_wr.num_sge = 1;
  read_wr.opcode = IBV_WR_RDMA_READ;
  read_wr.send_flags = IBV_SEND_SIGNALED;
  read_wr.wr.rdma.remote_addr = remote_addr;
  read_wr.wr.rdma.rkey = rkey;

  ibv_send_wr* bad_wr = nullptr;
  int ret = ibv_post_send(ctx->qp, &read_wr, &bad_wr);
  if (ret != 0) {
    ERR("RDMA Read verification failed: %s (errno: %d)",
        strerror(errno),
        errno);
    return false;
  }

  if (!poll_cq_with_timeout(ctx, RDMA_POLL_CQE_TIMEOUT, 1)) {
    ERR("RDMA Read verification polling failed");
    return false;
  }

  if (KVCacheConfig::getInstance().is_debug_output_enabled()) {
    uint8_t* data = reinterpret_cast<uint8_t*>(ctx->conn.read_bufs[block_idx]);
    uint8_t first_byte = data[0];
    uint8_t last_byte = data[block_size_byte - 1];
    DEBUG(
        "Read verification success - Block %zu (Layer: %d, %s:%s): first=%u, "
        "last=%u",
        block_idx,
        layer_idx,
        ip.c_str(),
        port.c_str(),
        static_cast<uint32_t>(first_byte),
        static_cast<uint32_t>(last_byte));
  }

  return true;
}
