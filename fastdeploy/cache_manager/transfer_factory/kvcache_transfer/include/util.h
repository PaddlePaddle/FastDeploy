#ifndef KVCACHE_UTILS_H
#define KVCACHE_UTILS_H

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "log.h"

#define PATH_MAX 4096 /* # chars in a path name including nul */
#define RDMA_WR_LIST_MAX_SIZE 32
#define RDMA_SQ_MAX_SIZE 1024

#define RDMA_DEFAULT_PORT 20001
#define RDMA_TCP_CONNECT_SIZE 1024
#define RDMA_POLL_CQE_TIMEOUT 30

/// @brief Connection status enumeration
enum class ConnStatus {
  kConnected,         // Connection is active
  kDisconnected,      // Connection is not active
  kError,             // Connection error occurred
  kTimeout,           // Connection timed out
  kInvalidParameters  // Invalid connection parameters
};

/// @brief Queue Pair (QP) setup result status
enum class QpStatus {
  kSuccess,            // Successfully transitioned QP to RTS
  kInvalidParameters,  // ctx or dest is null
  kDeviceQueryFailed,  // ibv_query_device failed
  kPortQueryFailed,    // ibv_query_port failed
  kMtuMismatch,        // Requested MTU exceeds active MTU
  kModifyToRTRFailed,  // Failed to modify QP to RTR
  kModifyToRTSFailed   // Failed to modify QP to RTS
};

/**
 * @brief Convert PCI bus ID string to int64_t
 * @param busId PCI bus ID string (e.g. "0000:3b:00.0")
 * @param[out] id Converted numeric ID
 */
inline void busid_to_int64(const char* busId, int64_t* id) {
  char hexStr[17] = {0};
  int hexOffset = 0;

  // Filter valid hex characters
  for (int i = 0; hexOffset < sizeof(hexStr) - 1 && busId[i] != '\0'; i++) {
    char c = busId[i];
    if (c == '.' || c == ':') continue;

    if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') ||
        (c >= 'a' && c <= 'f')) {
      hexStr[hexOffset++] = c;
    }
  }

  *id = strtol(hexStr, NULL, 16);
}

class NetworkInterfaceManager {
 public:
  struct InterfaceInfo {
    std::string name;
    std::string ip;
    bool is_up;
    bool is_running;
    bool is_loopback;

    bool isUsable() const { return is_up && is_running && !is_loopback; }
  };

  static std::vector<InterfaceInfo> getAllInterfaces() {
    std::vector<InterfaceInfo> interfaces;
    struct ifaddrs* ifaddrs_ptr = nullptr;

    if (getifaddrs(&ifaddrs_ptr) == -1) {
      return interfaces;
    }

    for (struct ifaddrs* ifa = ifaddrs_ptr; ifa != nullptr;
         ifa = ifa->ifa_next) {
      if (ifa->ifa_addr == nullptr) continue;
      if (ifa->ifa_addr->sa_family != AF_INET) continue;

      InterfaceInfo info;
      info.name = ifa->ifa_name;
      info.is_up = (ifa->ifa_flags & IFF_UP) != 0;
      info.is_running = (ifa->ifa_flags & IFF_RUNNING) != 0;
      info.is_loopback = (ifa->ifa_flags & IFF_LOOPBACK) != 0;

      struct sockaddr_in* sa = (struct sockaddr_in*)ifa->ifa_addr;
      char ip_str[INET_ADDRSTRLEN];
      inet_ntop(AF_INET, &sa->sin_addr, ip_str, INET_ADDRSTRLEN);
      info.ip = ip_str;

      interfaces.push_back(info);
    }

    freeifaddrs(ifaddrs_ptr);
    return interfaces;
  }

  static std::string getFirstUsableInterface() {
    auto interfaces = getAllInterfaces();

    for (const auto& iface : interfaces) {
      if (iface.isUsable()) {
        return iface.name;
      }
    }
    return "";
  }

  static void displayAllInterfaces() {
    auto interfaces = getAllInterfaces();

    printf("Available network interfaces:\n");
    for (const auto& iface : interfaces) {
      printf("  %s: %s [%s%s%s]\n",
             iface.name.c_str(),
             iface.ip.c_str(),
             iface.is_up ? "UP" : "DOWN",
             iface.is_running ? ",RUNNING" : "",
             iface.is_loopback ? ",LOOPBACK" : "");
    }
  }
};

class KVCacheConfig {
 private:
  // Configuration values
  int rdma_gid_index_;
  bool has_rdma_dest_port_override_;  // 替代 std::optional
  int rdma_dest_port_override_;
  const char* socket_interface_;
  char* socket_interface_buffer_;
  bool gdrcopy_flush_enabled_;
  bool verify_read_enabled_;
  bool debug_mode_enabled_;
  bool debug_output_enabled_;
  const char* debug_file_path_;
  const char* error_file_path_;
  bool relax_ordering_enabled_;
  int ib_timeout_;
  const char* rdma_nics_;

  // Private constructor for singleton pattern
  KVCacheConfig() {
    // Initialize configuration from environment variables
    rdma_gid_index_ = parse_int_value(
        std::getenv("KVCACHE_RDMA_GID_INDEX"), 3, "KVCACHE_RDMA_GID_INDEX");

    // Parse optional RDMA port override
    const char* port_value = std::getenv("SET_RDMA_DEST_PORT");
    has_rdma_dest_port_override_ = false;  // 默认为false
    if (port_value) {
      try {
        rdma_dest_port_override_ = std::stoi(std::string(port_value));
        has_rdma_dest_port_override_ = true;
      } catch (const std::exception& e) {
        fprintf(stderr,
                "Invalid SET_RDMA_DEST_PORT value: '%s', ignoring\n",
                port_value);
      }
    }

    const char* env_interface = std::getenv("KVCACHE_SOCKET_IFNAME");

    if (env_interface && env_interface[0] != '\0') {
      socket_interface_ = env_interface;
      printf("Using specified interface: %s\n", socket_interface_);
    } else {
      std::string iface = NetworkInterfaceManager::getFirstUsableInterface();
      if (!iface.empty()) {
        socket_interface_buffer_ = new char[iface.size() + 1];
        std::strcpy(socket_interface_buffer_, iface.c_str());
        socket_interface_ = socket_interface_buffer_;
        printf("Auto-detected interface: %s\n", socket_interface_);
      } else {
        fprintf(stderr, "Warning: No usable network interface found\n");
        socket_interface_ = "";
      }
      NetworkInterfaceManager::displayAllInterfaces();
    }

    socket_interface_ = std::getenv("KVCACHE_SOCKET_IFNAME");
    debug_file_path_ = std::getenv("KVCACHE_DEBUG_FILE");
    error_file_path_ = std::getenv("KVCACHE_ERROR_FILE");

    gdrcopy_flush_enabled_ =
        parse_bool_value(std::getenv("KVCACHE_GDRCOPY_FLUSH_ENABLE"));
    verify_read_enabled_ = parse_bool_value(std::getenv("KVCACHE_VERIFY_READ"));
    debug_mode_enabled_ = parse_bool_value(std::getenv("KVCACHE_DEBUG")) ||
                          parse_bool_value(std::getenv("KV_IS_DEBUG_ENABLED"));
    debug_output_enabled_ =
        parse_bool_value(std::getenv("KVCACHE_DEBUG_OUTPUT"));

    relax_ordering_enabled_ =
        parse_bool_value(std::getenv("KVCACHE_RELAX_ORDERING"));

    ib_timeout_ = parse_int_value(
        std::getenv("KVCACHE_IB_TIMEOUT"), 18, "KVCACHE_IB_TIMEOUT");

    rdma_nics_ = std::getenv("KVCACHE_RDMA_NICS");
  }

  // Helper methods
  bool parse_bool_value(const char* value) {
    if (!value) return false;

    std::string str_value(value);
    std::transform(
        str_value.begin(), str_value.end(), str_value.begin(), ::tolower);

    return (str_value == "1" || str_value == "true" || str_value == "on" ||
            str_value == "yes");
  }

  int parse_int_value(const char* value,
                      int default_value,
                      const char* env_name) {
    if (!value) return default_value;

    try {
      return std::stoi(std::string(value));
    } catch (const std::invalid_argument& e) {
      fprintf(stderr,
              "Invalid value for %s: '%s', using default: %d\n",
              env_name,
              value,
              default_value);
      return default_value;
    } catch (const std::out_of_range& e) {
      fprintf(stderr,
              "%s value out of range: '%s', using default: %d\n",
              env_name,
              value,
              default_value);
      return default_value;
    }
  }

 public:
  // Prevent copying and assignment
  KVCacheConfig(const KVCacheConfig&) = delete;
  KVCacheConfig& operator=(const KVCacheConfig&) = delete;

  // Get singleton instance
  static KVCacheConfig& getInstance() {
    static KVCacheConfig instance;
    return instance;
  }

  int get_ib_timeout() const { return ib_timeout_; }

  // Configuration retrieval methods
  int get_rdma_gid_index() const { return rdma_gid_index_; }

  int resolve_rdma_dest_port(int default_port) const {
    return has_rdma_dest_port_override_ ? rdma_dest_port_override_
                                        : default_port;
  }

  int resolve_rdma_dest_port(const std::string& default_port) const {
    try {
      return resolve_rdma_dest_port(std::stoi(default_port));
    } catch (const std::exception& e) {
      fprintf(
          stderr, "Invalid default port string: %s\n", default_port.c_str());
      return 0;
    }
  }

  const char* get_socket_interface() const { return socket_interface_; }
  const char* get_debug_file_path() const { return debug_file_path_; }
  const char* get_error_file_path() const { return error_file_path_; }
  const char* get_rdma_nics() const { return rdma_nics_; }

  // Feature check methods
  bool is_gdrcopy_flush_enabled() const { return gdrcopy_flush_enabled_; }
  bool is_verify_read_enabled() const { return verify_read_enabled_; }
  bool is_debug_mode_enabled() const { return debug_mode_enabled_; }
  bool is_debug_output_enabled() const { return debug_output_enabled_; }
  bool is_relax_ordering_enabled() const { return relax_ordering_enabled_; }

  // Display configuration
  void displayConfiguration() const {
    INFO("KVCache Configuration:\n");
    INFO("Init KVCacheConfig RDMA GID Index: %d\n", rdma_gid_index_);

    if (has_rdma_dest_port_override_) {
      INFO("Init KVCacheConfig RDMA Destination Port Override: %d\n",
           rdma_dest_port_override_);
    }

    if (socket_interface_) {
      INFO("Init KVCacheConfig  Socket Interface: %s\n", socket_interface_);
    }

    INFO("Init KVCacheConfig GDRCopy Flush: %s\n",
         gdrcopy_flush_enabled_ ? "enabled" : "disabled");
    INFO("Init KVCacheConfig Verify Read: %s\n",
         verify_read_enabled_ ? "enabled" : "disabled");
    INFO("Init KVCacheConfig Debug Mode: %s\n",
         debug_mode_enabled_ ? "enabled" : "disabled");
    INFO("Init KVCacheConfig Debug Output: %s\n",
         debug_output_enabled_ ? "enabled" : "disabled");

    if (debug_file_path_) {
      INFO("Init KVCacheConfig Debug File: %s\n", debug_file_path_);
    }

    if (error_file_path_) {
      INFO("Init KVCacheConfig Error File: %s\n", error_file_path_);
    }
  }
};

#endif
