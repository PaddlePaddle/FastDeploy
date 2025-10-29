/**
 * @file kvcache_connection.cpp
 * @brief RDMA connection implementation for key-value cache
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

#include "kvcache_connection.h"
#include "log.h"  // For logging system

// Global variables
pthread_mutex_t g_ib_lock = PTHREAD_MUTEX_INITIALIZER;
std::vector<IbDeviceInfo> g_ib_all_devs;

/**
 * @brief Get IB device PCI bus ID as int64_t
 * @param dev_name InfiniBand device name
 * @return PCI bus ID as int64_t, -1 on error
 */
static int64_t get_ib_busid(const char *dev_name) {
  char dev_path[PATH_MAX];
  snprintf(dev_path, PATH_MAX, "/sys/class/infiniband/%s/device", dev_name);

  char *p = realpath(dev_path, NULL);
  if (p == NULL) {
    WARN("Failed to get realpath for device %s: %s", dev_name, strerror(errno));
    return -1;
  }

  // Extract bus ID from path
  int offset = strlen(p) - 1;
  while (offset >= 0 && p[offset] != '/') {
    offset--;
  }

  if (offset < 0) {
    free(p);
    return -1;
  }

  char bus_str[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  strncpy(bus_str, p + offset + 1, sizeof(bus_str) - 1);
  bus_str[sizeof(bus_str) - 1] = '\0';
  free(p);

  int64_t ret;
  busid_to_int64(bus_str, &ret);
  return ret;
}

/**
 * @brief Parse and cache IB device information
 * @return Number of IB devices found, negative on error
 *
 * @note This function is thread-safe and will only parse once
 */
int parse_port_ib_info() {
  if (g_kvcache_ib_dev_nums != -1) return 0;

  pthread_mutex_lock(&g_ib_lock);
  if (g_kvcache_ib_dev_nums != -1) {
    pthread_mutex_unlock(&g_ib_lock);
    return 0;
  }

  INFO("Initializing IB device information");
  g_kvcache_ib_dev_nums = 0;

  const char *env_nics = KVCacheConfig::getInstance().get_rdma_nics();
  if (!env_nics) {
    ERR("Environment variable KVCACHE_RDMA_NICS not set");
    pthread_mutex_unlock(&g_ib_lock);
    return -1;
  }

  // Parse NIC list
  char nic_names[MAXNAMESIZE][MAXNAMESIZE] = {0};
  int nic_count = 0;
  char *env_copy = strdup(env_nics);
  if (!env_copy) {
    ERR("Failed to duplicate NIC list string");
    pthread_mutex_unlock(&g_ib_lock);
    return -1;
  }

  for (char *token = strtok(env_copy, ","); token && nic_count < MAXNAMESIZE;
       token = strtok(NULL, ",")) {
    strncpy(nic_names[nic_count++], token, MAXNAMESIZE - 1);
  }
  free(env_copy);

  // Get IB device list
  int total_devs = 0;
  ibv_device **dev_list = ibv_get_device_list(&total_devs);
  if (!dev_list || total_devs <= 0) {
    ERR("No IB devices found, ibv_get_device_list failed, total_devs = %d",
        total_devs);
    pthread_mutex_unlock(&g_ib_lock);
    return -1;
  }
  INFO("Found %d IB devices, filtering by NIC list", total_devs);

  for (int i = 0;
       i < total_devs && g_kvcache_ib_dev_nums < KVCACHE_RDMA_MAX_NICS;
       ++i) {
    const char *dev_name = dev_list[i]->name;

    bool allowed = false;
    for (int j = 0; j < nic_count; ++j) {
      if (strcmp(dev_name, nic_names[j]) == 0) {
        allowed = true;
        break;
      }
    }
    if (!allowed) {
      WARN("Skipping device not in NIC list: %s", dev_name);
      continue;
    }

    ibv_context *ctx = ibv_open_device(dev_list[i]);
    if (!ctx) {
      ERR("Failed to open device %s: %s", dev_name, strerror(errno));
      continue;
    }

    ibv_device_attr dev_attr = {};
    if (ibv_query_device(ctx, &dev_attr) != 0) {
      ERR("Failed to query device %s: %s", dev_name, strerror(errno));
      ibv_close_device(ctx);
      continue;
    }

    int valid_ports = 0;
    for (int port_num = 1; port_num <= dev_attr.phys_port_cnt; ++port_num) {
      ibv_port_attr port_attr = {};
      if (ibv_query_port(ctx, port_num, &port_attr) != 0) {
        WARN("Failed to query port %d on device %s: %s",
             port_num,
             dev_name,
             strerror(errno));
        continue;
      }

      if (port_attr.state != IBV_PORT_ACTIVE) {
        WARN("Port %d on device %s is not active (state: %d)",
             port_num,
             dev_name,
             port_attr.state);
        continue;
      }

      if (port_attr.link_layer != IBV_LINK_LAYER_INFINIBAND &&
          port_attr.link_layer != IBV_LINK_LAYER_ETHERNET) {
        WARN("Unsupported link layer %d on device %s port %d",
             port_attr.link_layer,
             dev_name,
             port_num);
        continue;
      }

      IbDeviceInfo dev_info = {};
      dev_info.device = i;
      dev_info.guid = dev_attr.sys_image_guid;
      dev_info.port = port_num;
      dev_info.link = port_attr.link_layer;
      dev_info.active_mtu = port_attr.active_mtu;
      dev_info.context = ctx;
      dev_info.busid = get_ib_busid(dev_name);
      dev_info.maxQp = dev_attr.max_qp;
      strncpy(dev_info.devName, dev_name, MAXNAMESIZE);

      INFO("Adding device %s port %d (%s)",
           dev_name,
           port_num,
           port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE");

      g_ib_all_devs.push_back(dev_info);
      ++g_kvcache_ib_dev_nums;
      ++valid_ports;
    }

    if (valid_ports == 0) {
      ERR("No valid ports found for device %s", dev_name);
      ibv_close_device(ctx);
    }
  }

  ibv_free_device_list(dev_list);
  INFO("Initialized %d IB devices", g_kvcache_ib_dev_nums);
  pthread_mutex_unlock(&g_ib_lock);
  return 0;
}

static int modify_qp_to_init(struct ibv_qp *qp, struct ibv_qp_attr *attr) {
  int ret = ibv_modify_qp(
      qp,
      attr,
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
  if (ret != 0) {
    ERR("Failed to modify QP to INIT: %s (errno=%d)", strerror(errno), errno);
  }

  return ret;
}

static int modify_qp_to_rtr(struct ibv_qp *qp, struct ibv_qp_attr *attr) {
  int ret = ibv_modify_qp(qp,
                          attr,
                          IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                              IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                              IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
  if (ret != 0) {
    ERR("Failed to modify QP to RTR: %s (errno=%d)", strerror(errno), errno);
  }

  return ret;
}

static int modify_rtr_to_rts(struct ibv_qp *qp, struct ibv_qp_attr *attr) {
  int ret = ibv_modify_qp(qp,
                          attr,
                          IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                              IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                              IBV_QP_MAX_QP_RD_ATOMIC);
  if (ret != 0) {
    ERR("Failed to modify QP to RTS: %s (errno=%d)", strerror(errno), errno);
  }
  return ret;
}

int server_exchange_qp_info(int connfd, QpInfo *local_dest, QpInfo *rem_dest) {
  if (!local_dest || !rem_dest) {
    ERR("Null pointer passed to server_exchange_qp_info");
    return -1;
  }

  char buffer[QpInfo::size];
  memset(buffer, 0, sizeof(buffer));

  // Read remote QP info from the connection
  int n = read(connfd, buffer, QpInfo::size);
  if (n != static_cast<int>(QpInfo::size)) {
    ERR("Failed to read remote QP info: read %d bytes, expected %zu",
        n,
        QpInfo::size);
    return -1;
  }

  QpInfo remote_msg;
  remote_msg.deserialize(buffer);
  *rem_dest = remote_msg;
  rem_dest->psn = 0;

  // Prepare local QP info to send
  QpInfo local_msg = *local_dest;
  local_msg.psn = 0;
  local_msg.serialize(buffer);

  // Send local QP info to the remote side
  n = write(connfd, buffer, QpInfo::size);
  if (n != static_cast<int>(QpInfo::size)) {
    ERR("Failed to send local QP info: wrote %d bytes", n);
    return -1;
  }

  return 0;
}

int get_port_info(struct ibv_context *Context,
                  int port,
                  struct ibv_port_attr *attr) {
  return ibv_query_port(Context, port, attr);
}

QpStatus modify_qp_to_rts(struct RdmaContext *ctx,
                          int port,
                          int my_psn,
                          struct QpInfo *dest,
                          int sgid_id) {
  if (!ctx || !dest) {
    ERR("Invalid input parameters: ctx or dest is NULL");
    return QpStatus::kInvalidParameters;
  }

  struct ibv_device_attr dev_attr;
  if (ibv_query_device(ctx->context, &dev_attr)) {
    ERR("Failed to query device attributes: %s (errno=%d)",
        strerror(errno),
        errno);
    return QpStatus::kDeviceQueryFailed;
  }

  struct ibv_port_attr port_attr;
  if (ibv_query_port(ctx->context, port, &port_attr)) {
    ERR("Failed to query port attributes: %s (errno=%d)",
        strerror(errno),
        errno);
    return QpStatus::kPortQueryFailed;
  }

  if (dest->mtu > port_attr.active_mtu) {
    ERR("Specified MTU (%d) is greater than active port MTU (%d)",
        dest->mtu,
        port_attr.active_mtu);
    return QpStatus::kMtuMismatch;
  }

  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(struct ibv_qp_attr));

  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = dest->mtu;
  attr.dest_qp_num = dest->qpn;
  attr.rq_psn = 0;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;

  attr.ah_attr.is_global = 1;
  attr.ah_attr.grh.hop_limit = 255;
  attr.ah_attr.grh.flow_label = 0;
  attr.ah_attr.grh.traffic_class = 0;
  attr.ah_attr.grh.dgid.global.subnet_prefix = (dest->gid.global.subnet_prefix);
  attr.ah_attr.grh.dgid.global.interface_id = (dest->gid.global.interface_id);
  attr.ah_attr.grh.sgid_index = sgid_id;

  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = port;

  if (modify_qp_to_rtr(ctx->qp, &attr) != 0) {
    return QpStatus::kModifyToRTRFailed;
  }

  int qp_timeout = KVCacheConfig::getInstance().get_ib_timeout();
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = qp_timeout;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.sq_psn = 0;
  attr.max_rd_atomic = 1;

  if (modify_rtr_to_rts(ctx->qp, &attr) != 0) {
    return QpStatus::kModifyToRTSFailed;
  }

  LOGD("QP successfully transitioned to RTS state");
  return QpStatus::kSuccess;
}

static std::shared_ptr<QpInfo> client_exch_dest(struct RdmaContext *ctx,
                                                const std::string &dst_ip,
                                                int port,
                                                const QpInfo *my_dest) {
  struct addrinfo hints = {};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  struct addrinfo *res = nullptr;
  std::ostringstream service;
  service << port;

  int ret = getaddrinfo(dst_ip.c_str(), service.str().c_str(), &hints, &res);
  if (ret != 0) {
    ERR("getaddrinfo failed for %s:%d - %s",
        dst_ip.c_str(),
        port,
        gai_strerror(ret));
    return nullptr;
  }

  int sockfd = -1;
  for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
    sockfd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (sockfd < 0) {
      WARN("Socket creation failed: %s", strerror(errno));
      continue;
    }

    int enable = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_KEEPALIVE, &enable, sizeof(enable));
    int keep_idle = 10, keep_intvl = 5, keep_cnt = 3;
    setsockopt(sockfd, SOL_TCP, TCP_KEEPIDLE, &keep_idle, sizeof(keep_idle));
    setsockopt(sockfd, SOL_TCP, TCP_KEEPINTVL, &keep_intvl, sizeof(keep_intvl));
    setsockopt(sockfd, SOL_TCP, TCP_KEEPCNT, &keep_cnt, sizeof(keep_cnt));

    if (connect(sockfd, ai->ai_addr, ai->ai_addrlen) == 0) {
      break;  // Connected
    }

    WARN("Connect failed: %s", strerror(errno));
    close(sockfd);
    sockfd = -1;
  }

  freeaddrinfo(res);

  if (sockfd < 0) {
    ERR("Unable to connect to %s:%d", dst_ip.c_str(), port);
    return nullptr;
  }

  ctx->sock_fd = sockfd;

  char buffer[QpInfo::size] = {};
  QpInfo(*my_dest).serialize(buffer);

  if (write(sockfd, buffer, QpInfo::size) != QpInfo::size) {
    WARN("Failed to send local QP info to %s", dst_ip.c_str());
    close(sockfd);
    return nullptr;
  }

  if (read(sockfd, buffer, QpInfo::size) != QpInfo::size) {
    WARN("Failed to receive remote QP info from %s", dst_ip.c_str());
    close(sockfd);
    return nullptr;
  }

  // I think no need to check memory allocate, because once allocate failed,
  // that's mean the process encountering OOM, let it crash then check whether
  // the code logic has memory leak or not.
  auto rem_dest = std::make_shared<QpInfo>();
  rem_dest->deserialize(buffer);
  return rem_dest;
}

bool poll_cq_with_timeout(struct RdmaContext *ctx,
                          int timeout_seconds,
                          int cqe_count) {
  struct timespec start_time, current_time;
  struct ibv_wc *wc_array =
      (struct ibv_wc *)malloc(cqe_count * sizeof(struct ibv_wc));

  if (!wc_array) {
    ERR("Failed to allocate memory for WC array");
    return false;
  }

  clock_gettime(CLOCK_MONOTONIC, &start_time);

  while (1) {
    int poll_result = ibv_poll_cq(ctx->cq, cqe_count, wc_array);

    if (poll_result < 0) {
      ERR("ibv_poll_cq failed with return value %d", poll_result);
      free(wc_array);
      return false;
    } else if (poll_result > 0) {
      for (int i = 0; i < poll_result; ++i) {
        if (wc_array[i].status == IBV_WC_SUCCESS) {
          LOGD("Work completion %d successful", poll_result);
        } else {
          LOGD("Work completion %d status is %d (%s)",
               poll_result,
               wc_array[i].status,
               ibv_wc_status_str(wc_array[i].status));
        }
      }
      free(wc_array);
      return true;
    }

    clock_gettime(CLOCK_MONOTONIC, &current_time);
    if ((current_time.tv_sec - start_time.tv_sec) >= timeout_seconds) {
      ERR("Timeout occurred after %d seconds", timeout_seconds);
      free(wc_array);
      return false;
    }
  }
  return true;
}

bool clear_qp_info(struct RdmaContext *ctx) {
  if (!ctx) {
    ERR("RdmaContext pointer is null.");
    return false;
  }

  bool success = true;

  if (ctx->qp) {
    if (ibv_destroy_qp(ctx->qp)) {
      ERR("Failed to destroy QP.");
      success = false;
    }
  }

  if (ctx->cq) {
    if (ibv_destroy_cq(ctx->cq)) {
      ERR("Failed to deallocate cq Domain.");
      success = false;
    }
  }

  if (ctx->channel) {
    if (ibv_destroy_comp_channel(ctx->channel)) {
      ERR("Failed to destroy Completion Channel.");
      success = false;
    }
  }

  return success;
}

struct RdmaContext *create_qp(struct IbDeviceInfo *ib_dev,
                              struct ibv_pd **g_pd) {
  struct RdmaContext *ctx = new RdmaContext();
  memset(ctx, 0, sizeof(struct RdmaContext));
  struct ibv_qp_init_attr qpInitAttr = {};
  ctx->context = ib_dev->context;

  if (*g_pd == NULL) {
    *g_pd = ibv_alloc_pd(ctx->context);
    if (*g_pd == NULL) {
      ERR("failed to allocate protection domain");
      free(ctx->context);
      return NULL;
    }
  }
  ctx->pd = *g_pd;

  // Create completion channel
  ctx->channel = ibv_create_comp_channel(ctx->context);
  if (!ctx->channel) {
    ERR("Failed to create completion channel: %s", strerror(errno));
    delete ctx;
    return NULL;
  }

  // Create completion queue
  ctx->cq = ibv_create_cq(ctx->context, 4096, ctx, ctx->channel, 0);
  if (!ctx->cq) {
    ERR("Failed to create completion queue: %s", strerror(errno));
    ibv_destroy_comp_channel(ctx->channel);
    delete ctx;
    return NULL;
  }

  // Request completion notifications
  if (ibv_req_notify_cq(ctx->cq, 0)) {
    ERR("Failed to request CQ notifications: %s", strerror(errno));
    ibv_destroy_cq(ctx->cq);
    ibv_destroy_comp_channel(ctx->channel);
    delete ctx;
    return NULL;
  }

  // Initialize QP attributes
  qpInitAttr.send_cq = ctx->cq;
  qpInitAttr.recv_cq = ctx->cq;
  qpInitAttr.qp_type = IBV_QPT_RC;
  qpInitAttr.cap.max_send_wr = 4096;
  qpInitAttr.cap.max_recv_wr = 4096;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data = 0;

  // Create queue pair
  ctx->qp = ibv_create_qp(ctx->pd, &qpInitAttr);
  if (!ctx->qp) {
    ERR("Failed to create queue pair: %s", strerror(errno));
    ibv_destroy_cq(ctx->cq);
    ibv_destroy_comp_channel(ctx->channel);
    delete ctx;
    return NULL;
  }

  // Modify QP to INIT state
  struct ibv_qp_attr qpAttr = {};
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = 0;
  qpAttr.port_num = 1;
  qpAttr.qp_access_flags =
      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;

  int ret = modify_qp_to_init(ctx->qp, &qpAttr);
  if (ret != 0) {
    ERR("Failed to modify QP to INIT state: %s (ret=%d)", strerror(errno), ret);
    ibv_destroy_qp(ctx->qp);
    ibv_destroy_cq(ctx->cq);
    ibv_destroy_comp_channel(ctx->channel);
    delete ctx;
    return NULL;
  }

  INFO("Successfully created QP 0x%x on device %s",
       ctx->qp->qp_num,
       ib_dev->devName);

  return ctx;
}

/**
 * @brief Exchange destination information with remote peer
 * @param ctx RDMA context
 * @param ib_port IB port number
 * @param mtu Maximum transmission unit
 * @param port TCP port for connection
 * @param gidx GID index
 * @param dst_ip Destination IP address
 * @return true on success, false on failure
 */
bool client_exchange_destinations(struct RdmaContext *ctx,
                                  int ib_port,
                                  unsigned int port,
                                  int gidx,
                                  const std::string &dst_ip) {
  if (!ctx || !ctx->context || !ctx->qp) {
    ERR("Invalid RDMA context or QP not initialized");
    return false;
  }

  LOGD("Exchanging destination info with %s:%u", dst_ip.c_str(), port);

  // Get local QP information
  struct QpInfo my_dest = {};
  if (get_port_info(ctx->context, ib_port, &ctx->portinfo)) {
    ERR("Failed to get port info for port %d", ib_port);
    return false;
  }

  my_dest.lid = ctx->portinfo.lid;
  my_dest.mtu = ctx->portinfo.active_mtu;

  // Validate LID for InfiniBand
  if (ctx->portinfo.link_layer != IBV_LINK_LAYER_ETHERNET && !my_dest.lid) {
    ERR("Invalid LID 0x%04x for non-Ethernet link layer", my_dest.lid);
    return false;
  }

  // Get GID if specified
  if (gidx >= 0) {
    if (ibv_query_gid(ctx->context, ib_port, gidx, &my_dest.gid)) {
      ERR("Failed to query GID for index %d on port %d", gidx, ib_port);
      return false;
    }
  } else {
    memset(&my_dest.gid, 0, sizeof(my_dest.gid));
  }

  my_dest.qpn = ctx->qp->qp_num;
  my_dest.psn = lrand48() & 0xffffff;

  // Log local address info
  char gid_str[33] = {0};
  inet_ntop(AF_INET6, &my_dest.gid, gid_str, sizeof(gid_str));

  if (dst_ip.empty()) {
    ERR("Empty destination IP address");
    return false;
  }

  // Exchange destination info with remote
  auto rem_dest = client_exch_dest(ctx, dst_ip, port, &my_dest);
  if (!rem_dest) {
    ERR("Failed to exchange destination info with %s:%u", dst_ip.c_str(), port);
    return false;
  }

  LOGD("Remote address - LID: 0x%04x, QPN: 0x%06x, PSN: 0x%06x, Mtu: %u",
       rem_dest->lid,
       rem_dest->qpn,
       rem_dest->psn,
       rem_dest->mtu);

  // Modify QP to RTS state
  if (modify_qp_to_rts(ctx, ib_port, my_dest.psn, rem_dest.get(), gidx) !=
      QpStatus::kSuccess) {
    ERR("Failed to modify QP 0x%x to RTS state", ctx->qp->qp_num);
    return false;
  }

  LOGD("Successfully established connection to %s:%u", dst_ip.c_str(), port);

  return true;
}

/**
 * Helper function to exchange memory region information
 * @param ctx The RDMA context
 * @param data_list Vector containing the data to be exchanged
 * @param is_client True if this is the client side operation, false for server
 * @return true on success, false on failure
 */
template <typename T>
bool exchange_mr_vector(struct RdmaContext *ctx,
                        std::vector<T> &data_list,
                        bool is_client) {
  if (is_client) {
    return client_receive_memory_region(
        ctx, data_list.data(), data_list.size() * sizeof(T));
  } else {
    return server_send_memory_region(
        ctx, data_list.data(), data_list.size() * sizeof(T));
  }
}

/**
 * Exchange memory region information for the client
 * Client receives remote memory region information from the server
 * @param ctx The RDMA context
 * @return true on success, false on failure
 */
bool client_exchange_mr(struct RdmaContext *ctx) {
  LOGD("verb client exchange mr: start");

  if (ctx->conn.layer_number <= 0) {
    ERR("Invalid layer number: %d", ctx->conn.layer_number);
    return false;
  }

  auto layer_num = ctx->conn.layer_number;
  std::vector<void *> key_ptrs(layer_num);
  std::vector<uint32_t> key_rkeys(layer_num);
  std::vector<void *> val_ptrs(layer_num);
  std::vector<uint32_t> val_rkeys(layer_num);

  if (!exchange_mr_vector(ctx, key_ptrs, true)) return false;
  if (!exchange_mr_vector(ctx, key_rkeys, true)) return false;
  if (!exchange_mr_vector(ctx, val_ptrs, true)) return false;
  if (!exchange_mr_vector(ctx, val_rkeys, true)) return false;

  for (int i = 0; i < layer_num; ++i) {
    ctx->conn.write_cache_key_remote_ptr_list.push_back(key_ptrs[i]);
    ctx->conn.write_cache_key_remote_rkey_list.push_back(key_rkeys[i]);
    ctx->conn.write_cache_value_remote_ptr_list.push_back(val_ptrs[i]);
    ctx->conn.write_cache_value_remote_rkey_list.push_back(val_rkeys[i]);
  }
  return true;
}

/**
 * Exchange memory region information for the server
 * Server sends its memory region information to the client
 * @param ctx The RDMA context
 * @return true on success, false on failure
 */
bool server_exchange_mr(struct RdmaContext *ctx) {
  LOGD("verbs server exchange mr: start");

  if (ctx->conn.layer_number <= 0) {
    ERR("Invalid layer number: %d", ctx->conn.layer_number);
    return false;
  }

  auto layer_num = ctx->conn.layer_number;
  auto &key_mrs = ctx->conn.write_cache_key_server_mr_list;
  auto &val_mrs = ctx->conn.write_cache_value_server_mr_list;

  // Verify that server memory regions are properly initialized
  if (key_mrs.size() != layer_num || val_mrs.size() != layer_num) {
    ERR("server write cache memory region size error");
    return false;
  }

  // Prepare memory region information to send
  std::vector<uint64_t> send_key_ptrs;
  std::vector<uint32_t> send_key_rkeys;
  std::vector<uint64_t> send_val_ptrs;
  std::vector<uint32_t> send_val_rkeys;

  send_key_ptrs.reserve(layer_num);
  send_key_rkeys.reserve(layer_num);
  send_val_ptrs.reserve(layer_num);
  send_val_rkeys.reserve(layer_num);

  // Collect memory region information from local MRs
  for (int i = 0; i < layer_num; ++i) {
    send_key_ptrs.push_back(reinterpret_cast<uint64_t>(key_mrs[i]->addr));
    send_key_rkeys.push_back(key_mrs[i]->rkey);
    send_val_ptrs.push_back(reinterpret_cast<uint64_t>(val_mrs[i]->addr));
    send_val_rkeys.push_back(val_mrs[i]->rkey);
  }

  // Send all vectors to client
  if (!exchange_mr_vector(ctx, send_key_ptrs, false)) return false;
  if (!exchange_mr_vector(ctx, send_key_rkeys, false)) return false;
  if (!exchange_mr_vector(ctx, send_val_ptrs, false)) return false;
  if (!exchange_mr_vector(ctx, send_val_rkeys, false)) return false;

  return true;
}

/**
 * Send memory region information from server to client
 *
 * @param ctx The RDMA context
 * @param local_mr Pointer to the local memory region to be sent
 * @param byte_num Size of the memory region in bytes
 * @return true on success, false on failure
 */
bool server_send_memory_region(struct RdmaContext *ctx,
                               void *local_mr,
                               int byte_num) {
  // Register the memory region for sending
  ctx->conn.send_mr = ibv_reg_mr(ctx->pd, local_mr, byte_num, 0);
  if (ctx->conn.send_mr == NULL) {
    ERR("ibv_reg_mr failed");
    return false;
  }

  // Prepare the send work request
  struct ibv_send_wr wr, *bad_wr = NULL;
  struct ibv_sge sge;

  memset(&wr, 0, sizeof(wr));
  wr.wr_id = reinterpret_cast<uintptr_t>(&ctx->conn);
  wr.opcode = IBV_WR_SEND;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.send_flags = IBV_SEND_SIGNALED;

  // Set up scatter-gather element
  sge.addr = (uintptr_t)local_mr;
  sge.length = byte_num;
  sge.lkey = ctx->conn.send_mr->lkey;

  // Post the send request
  int ret = ibv_post_send(ctx->qp, &wr, &bad_wr);
  if (ret) {
    ERR("ibv_post_send failed");
    ibv_dereg_mr(ctx->conn.send_mr);
    return false;
  }

  // Wait for completion
  struct ibv_wc wc;
  ctx->conn.wc_count = 0;
  ctx->conn.wc_target_count = 0;

  if (!poll_cq_with_timeout(ctx, RDMA_POLL_CQE_TIMEOUT, 1)) {
    return false;
  }

  // Deregister the memory region
  ibv_dereg_mr(ctx->conn.send_mr);
  return true;
}

/**
 * Receive memory region information on the client side
 *
 * @param ctx The RDMA context
 * @param remote_mr Pointer to the buffer where remote memory region info will
 * be stored
 * @param byte_num Size of the memory region in bytes
 * @return true on success, false on failure
 */
bool client_receive_memory_region(struct RdmaContext *ctx,
                                  void *remote_mr,
                                  int byte_num) {
  // Register memory region for receiving data
  int access_flags = IBV_ACCESS_LOCAL_WRITE;
  ctx->conn.recv_mr = ibv_reg_mr(ctx->pd, remote_mr, byte_num, access_flags);
  if (ctx->conn.recv_mr == NULL) {
    ERR("ibv_reg_mr failed for receive region");
    return false;
  }

  // Prepare the receive work request
  struct ibv_recv_wr wr, *bad_wr = NULL;
  struct ibv_sge sge;

  memset(&wr, 0, sizeof(wr));
  wr.wr_id = reinterpret_cast<uintptr_t>(&ctx->conn);
  wr.sg_list = &sge;
  wr.num_sge = 1;

  // Set up scatter-gather element
  sge.addr = (uintptr_t)remote_mr;
  sge.length = byte_num;
  sge.lkey = ctx->conn.recv_mr->lkey;

  // Post the receive request
  int ret = ibv_post_recv(ctx->qp, &wr, &bad_wr);
  if (ret) {
    ibv_dereg_mr(ctx->conn.recv_mr);
    return false;
  }

  // Poll completion queue with timeout
  ctx->conn.wc_count = 0;
  ctx->conn.wc_target_count = 0;
  if (!poll_cq_with_timeout(ctx, RDMA_POLL_CQE_TIMEOUT, 1)) {
    return false;
  }

  // Deregister memory region
  ibv_dereg_mr(ctx->conn.recv_mr);
  return true;
}

/**
 * Sets up a listening socket on the specified port
 *
 * @param port The port number to listen on
 * @return The socket file descriptor on success, -1 on failure
 */
int setup_listening_socket(int port) {
  int sockfd = -1;
  struct addrinfo hints = {0};

  // Set up hints for getaddrinfo
  hints.ai_flags = AI_PASSIVE;
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  struct addrinfo *res = nullptr;

  // Convert port to string for getaddrinfo
  std::ostringstream service;
  service << port;

  // Get address info for the specified port
  int n = getaddrinfo(nullptr, service.str().c_str(), &hints, &res);
  if (n != 0) {
    ERR("getaddrinfo failed for port %d: %s", port, gai_strerror(n));
    return -1;
  }

  // Check if a specific network interface is specified
  const char *ifname = KVCacheConfig::getInstance().get_socket_interface();
  // Try each address until we successfully bind to one
  for (struct addrinfo *t = res; t; t = t->ai_next) {
    // Create socket
    sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
    if (sockfd < 0) {
      ERR("Socket creation failed: %s", strerror(errno));
      continue;
    }

    // Bind to specific interface if requested
    if (ifname) {
      WARN("Binding socket to the specified interface: %s", ifname);
      if (setsockopt(
              sockfd, SOL_SOCKET, SO_BINDTODEVICE, ifname, strlen(ifname)) <
          0) {
        ERR("Failed to bind to interface %s - %s", ifname, strerror(errno));
        close(sockfd);
        continue;
      }
    }

    // Enable address reuse
    n = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &n, sizeof(n));

    // Attempt to bind to the address
    if (bind(sockfd, t->ai_addr, t->ai_addrlen) == 0) {
      break;  // Successful bind
    } else {
      WARN("Bind failed: %s", strerror(errno));
      close(sockfd);
      sockfd = -1;
    }
  }

  // Free the address list
  freeaddrinfo(res);

  // Check if binding was successful
  if (sockfd < 0) {
    ERR("Couldn't bind to any address on port %d", port);
    return -1;
  }

  // Start listening for connections
  if (listen(sockfd, 4096) < 0) {
    ERR("Failed to listen on port %d: %s", port, strerror(errno));
    close(sockfd);
    return -1;
  }

  // Set socket to non-blocking mode
  int flags = fcntl(sockfd, F_GETFL, 0);
  int ret = fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);
  if (ret < 0) {
    ERR("Failed to set non-blocking mode on event channel");
    close(sockfd);
    return -1;
  }

  // Enable TCP keep-alive
  int enable = 1;
  if (setsockopt(sockfd, SOL_SOCKET, SO_KEEPALIVE, &enable, sizeof(enable)) <
      0) {
    ERR("Failed to enable TCP keep-alive on socket: %s", strerror(errno));
    close(sockfd);
    return -1;
  }

  return sockfd;
}

int configure_epoll(int sockfd) {
  int epollfd = epoll_create1(0);
  if (epollfd == -1) {
    ERR("epoll_create1");
  }

  // Initialize epoll for the listening socket
  struct epoll_event ev;
  ev.events = EPOLLIN | EPOLLOUT | EPOLLERR;
  ev.data.fd = sockfd;
  if (epoll_ctl(epollfd, EPOLL_CTL_ADD, sockfd, &ev) == -1) {
    ERR("Failed to add listening socket to epoll");
    close(sockfd);
    return -1;
  }

  return epollfd;
}

static char *get_ip_by_ifname(const char *ifname) {
  int fd = 0;
  struct ifreq ifr;
  struct sockaddr_in *ip_addr = NULL;

  fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (fd <= 0) {
    ERR("create socket failed: %s", strerror(errno));
    return NULL;
  }
  ifr.ifr_addr.sa_family = AF_INET;
  strncpy(ifr.ifr_name, ifname, IFNAMSIZ - 1);
  if (ioctl(fd, SIOCGIFADDR, &ifr) == 0) {
    ip_addr = (struct sockaddr_in *)&ifr.ifr_addr;
    close(fd);
    return inet_ntoa(ip_addr->sin_addr);
  } else {
    WARN("get ip from %s failed, error: %s", ifr.ifr_name, strerror(errno));
    close(fd);
    return NULL;
  }
}

std::vector<std::string> get_net_ifname() {
  std::vector<std::string> local_ip;
  char ifnames[KVCACHE_RDMA_NIC_MAX_LEN + 1] = {0};
  const char *tmp = KVCacheConfig::getInstance().get_socket_interface();
  if (tmp) {
    int cp_len = strlen(tmp) > KVCACHE_RDMA_NIC_MAX_LEN
                     ? KVCACHE_RDMA_NIC_MAX_LEN
                     : strlen(tmp);
    memcpy(ifnames, tmp, cp_len);
    ifnames[cp_len] = '\0';
  } else {
    WARN("no ifnames, local_ip: %lu", local_ip.size());
    return local_ip;
  }
  char *delim = (char *)",";
  int i = 0;
  WARN("ifnames: %s", ifnames);

  std::string rdma_addr[KVCACHE_RDMA_MAX_NICS];
  char *saveptr = nullptr;
  char *pch = strtok_r(ifnames, delim, &saveptr);
  while (pch != NULL) {
    rdma_addr[i++] = std::string(pch);
    pch = strtok_r(NULL, delim, &saveptr);
  }
  int dev_id = 0;
  while (dev_id < i) {
    if (rdma_addr[dev_id].length() != 0) {
      char *ip = get_ip_by_ifname(rdma_addr[dev_id].c_str());
      if (ip) {
        local_ip.push_back(std::string(ip));
      }
    }
    dev_id++;
  }
  return local_ip;
}

Connection::~Connection() {
  write_cache_key_server_mr_list.clear();
  write_cache_value_server_mr_list.clear();
  write_cache_key_remote_ptr_list.clear();
  write_cache_key_remote_rkey_list.clear();
  write_cache_value_remote_ptr_list.clear();
  write_cache_value_remote_rkey_list.clear();
  LOGD("delete Connection %s", url.c_str());
}
