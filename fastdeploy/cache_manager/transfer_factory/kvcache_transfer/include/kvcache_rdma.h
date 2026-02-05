#ifndef KVCACHE_RDMA_H
#define KVCACHE_RDMA_H

#pragma once

#include <rdma/rdma_cma.h>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include "kvcache_connection.h"
#include "log.h"
#include "util.h"  // Contains constant definitions

/**
 * @brief RDMA communication handler for key-value cache
 */
class RDMACommunicator {
 public:
  // Construction/Destruction
  RDMACommunicator(std::string& role,
                   int gpu_idx,
                   std::string& port,
                   std::vector<int64_t> local_key_cache,
                   std::vector<int64_t> local_value_cache,
                   int block_number,
                   int block_bytes,
                   std::vector<int64_t> local_key_cache_scale,
                   std::vector<int64_t> local_value_cache_scale,
                   int scale_block_bytes,
                   int prefill_tp_size,
                   int prefill_tp_idx);
  ~RDMACommunicator();

  // Connection management
  int connect(const std::string& dst_ip,
              const std::string& dst_port,
              int dest_tp_size);
  bool is_connected(const std::string& dst_ip, const std::string& dst_port);

  // Core functionality
  int write_cache(const std::string& ip,
                  const std::string& port,
                  const std::vector<int64_t>& local_block_ids,
                  const std::vector<int64_t>& remote_block_ids,
                  int32_t layer_idx);

  // Server Init
  int init_server();

  // get socket nic ip
  std::string fetch_local_ip();

 private:
  // Server Core functions
  int start_server(int sport, int sgid_idx, int gpu_index);

  // Internal implementation methods
  void resize_vectors();
  void assign_pointers();
  void validate_addr();
  bool client_mr_register_per_layer(struct RdmaContext* ctx);
  bool server_mr_register_per_layer(struct RdmaContext* ctx);
  struct ibv_mr* register_memory_region(ibv_pd* pd,
                                        void* addr,
                                        size_t size,
                                        const std::string& desc,
                                        uint32_t access_flags);
  bool deregister_memory_regions(struct RdmaContext* ctx);

  bool post_block_send(struct RdmaContext* ctx,
                       int layer_idx,
                       const std::vector<int64_t>& local_block_ids,
                       const std::string data_type,
                       std::vector<uint64_t>& remote_addr,
                       uint32_t rkey,
                       const std::string& ip,
                       const std::string& port);

  bool execute_rdma_writes(struct RdmaContext* ctx,
                           int layer_idx,
                           const std::vector<int64_t>& local_block_ids,
                           const std::string data_type,
                           std::vector<uint64_t>& remote_addr,
                           uint32_t rkey);

  void prepare_write_requests(struct ibv_sge* sge_list,
                              struct ibv_send_wr* send_wr_list,
                              int layer_idx,
                              const std::vector<int64_t>& local_block_ids,
                              const std::string data_type,
                              std::vector<uint64_t>& remote_addr,
                              uint32_t rkey);

  bool execute_read_verification(struct RdmaContext* ctx,
                                 size_t block_idx,
                                 uint64_t remote_addr,
                                 uint32_t rkey,
                                 int layer_idx,
                                 const std::string& ip,
                                 const std::string& port);

  bool post_send_with_retry(struct RdmaContext* ctx,
                            struct ibv_send_wr* wr_list,
                            size_t inflight_wr,
                            bool need_poll);

  // Connection management
  int client_listener();
  void close_server_connection(
      int fd,
      struct RdmaContext* ctx,
      int epollfd,
      std::map<int, struct RdmaContext*>& connectionContexts);
  void close_client_connection(int fd, struct RdmaContext* ctx, int epollfd);

  void remove_conn(const std::string& url);
  struct RdmaContext* get_conn(const std::string& ip, const std::string& port);

  // Member variables
  std::string splitwise_role;  // Role in distributed system ("decode" or other)
  int gpu_idx;                 // GPU device index
  std::string port;            // Communication port
  std::vector<int64_t> local_cache_key_ptr_layer_head_;  // Key cache pointers
  std::vector<int64_t>
      local_cache_value_ptr_layer_head_;  // Value cache pointers
  std::vector<int64_t>
      local_cache_key_scale_ptr_layer_head_;  // Key cache pointers
  std::vector<int64_t>
      local_cache_value_scale_ptr_layer_head_;  // Value cache pointers
  int block_number;                             // Number of blocks
  int block_size_byte;                          // Size of each block in bytes
  int scale_block_size_byte;  // Size of each scale block in bytes
  int layer_number;           // Number of layers
  int prefill_tp_size;        // tensor parallelism size for prefill
  int prefill_tp_idx;         // tensor parallelism index for prefill

  // The key and value cache pointers for each layer
  std::vector<std::vector<void*>> local_cache_key_ptr_per_layer;
  std::vector<std::vector<void*>> local_cache_value_ptr_per_layer;

  std::vector<std::vector<void*>> local_cache_key_scale_ptr_per_layer;
  std::vector<std::vector<void*>> local_cache_value_scale_ptr_per_layer;

  // Memory regions in client(prefill), only registered once for each layer
  std::vector<struct ibv_mr*> write_mr_key_list;
  std::vector<struct ibv_mr*> write_mr_value_list;
  std::vector<struct ibv_mr*> write_mr_key_scale_list;
  std::vector<struct ibv_mr*> write_mr_value_scale_list;

  // Memory regions in server(decode)
  std::vector<struct ibv_mr*> write_cache_key_server_mr_list;
  std::vector<struct ibv_mr*> write_cache_value_server_mr_list;
  std::vector<struct ibv_mr*> write_cache_key_scale_server_mr_list;
  std::vector<struct ibv_mr*> write_cache_value_scale_server_mr_list;

  std::vector<std::string> main_ip_list;  // List of local IP addresses
  std::map<std::string, struct RdmaContext*>
      conn_map;                        // Active connections map
  std::mutex mutex_;                   // Thread synchronization mutex
  int rdma_event_channel_epoll_fd;     // Epoll file descriptor
  struct ibv_pd* g_pd = NULL;          // fd
  int RDMACommunicator_status;         // Communicator status flag
  bool start_client_listener = false;  // Client listener flag

  bool has_value_cache_;  // MLA does not have value cache.
  bool has_key_scale_;
  bool has_value_scale_;
};

#endif  // KVCACHE_RDMA_H
