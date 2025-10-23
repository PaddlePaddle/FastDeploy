#pragma once

/**
 * @file log.h
 * @brief Logging module for key-value cache system
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

#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>  //for gethostname
#include <chrono>
#include <ctime>
#include <string>

#define KV_IS_DEBUG_ENABLED (std::getenv("KVCACHE_DEBUG"))
#define FILE_NAME(x) (strrchr(x, '/') ? strrchr(x, '/') + 1 : x)

static thread_local char __attribute__((__unused__)) str[64];

// for log levels (C++ enum class style in C)
typedef enum {
  KV_LOG_LEVEL_INFO = 0,
  KV_LOG_LEVEL_DEBUG = 1,
  KV_LOG_LEVEL_WARN = 2,
  KV_LOG_LEVEL_ERROR = 3
} KVLogLevel;

void debug_log(KVLogLevel level,
               bool enable_to_terminal,
               const char *filefunc,
               int line,
               const char *fmt,
               ...) __attribute__((format(printf, 5, 6)));

/**
 * @brief Unified logging macro to reduce duplication and improve
 * maintainability.
 *
 * @param level              Log level (e.g., INFO, DEBUG, WARN, ERR).
 * @param to_terminal        If true, the log will be printed to terminal.
 * @param ...                Format string and arguments (like printf).
 */
#define KV_LOG(level, to_terminal, ...) \
  debug_log(level, to_terminal, FILE_NAME(__FILE__), __LINE__, __VA_ARGS__)

// Public logging macros with terminal output
#define WARN(...) KV_LOG(KV_LOG_LEVEL_WARN, true, __VA_ARGS__)
#define ERR(...) KV_LOG(KV_LOG_LEVEL_ERROR, true, __VA_ARGS__)
#define DEBUG(...) KV_LOG(KV_LOG_LEVEL_DEBUG, true, __VA_ARGS__)
#define INFO(...) KV_LOG(KV_LOG_LEVEL_INFO, true, __VA_ARGS__)

#define gettid() ((pid_t)syscall(SYS_gettid))
#define GET_CURRENT_TIME()               \
  do {                                   \
    time_t timer = time(0);              \
    struct tm *t = localtime(&timer);    \
    char hostname[32];                   \
    gethostname(hostname, 32);           \
    sprintf(str,                         \
            "%02d:%02d:%02d][%.32s][%d", \
            t->tm_hour,                  \
            t->tm_min,                   \
            t->tm_sec,                   \
            hostname,                    \
            gettid());                   \
  } while (0)

#define LOGE(fmt, arg...)                           \
  do {                                              \
    GET_CURRENT_TIME();                             \
    fprintf(stderr,                                 \
            "[%s][ERR][KV_CACHE][%s:%d] " fmt "\n", \
            str,                                    \
            FILE_NAME(__FILE__),                    \
            __LINE__,                               \
            ##arg);                                 \
  } while (0)

#define LOGW(fmt, arg...)                            \
  do {                                               \
    GET_CURRENT_TIME();                              \
    fprintf(stderr,                                  \
            "[%s][WARN][KV_CACHE][%s:%d] " fmt "\n", \
            str,                                     \
            FILE_NAME(__FILE__),                     \
            __LINE__,                                \
            ##arg);                                  \
  } while (0)

#define LOGI(fmt, arg...)                            \
  do {                                               \
    GET_CURRENT_TIME();                              \
    fprintf(stdout,                                  \
            "[%s][INFO][KV_CACHE][%s:%d] " fmt "\n", \
            str,                                     \
            FILE_NAME(__FILE__),                     \
            __LINE__,                                \
            ##arg);                                  \
  } while (0)

#define LOGD(fmt, arg...)                             \
  do {                                                \
    if (KV_IS_DEBUG_ENABLED) {                        \
      GET_CURRENT_TIME();                             \
      fprintf(stdout,                                 \
              "[%s][DBG][KV_CACHE][%s:%d] " fmt "\n", \
              str,                                    \
              FILE_NAME(__FILE__),                    \
              __LINE__,                               \
              ##arg);                                 \
    }                                                 \
  } while (0)

#define LOGD_IF(cond, fmt, ...)         \
  do {                                  \
    if ((cond)) LOGD(fmt, __VA_ARGS__); \
  } while (0)

#define LOGD_RAW(fmt, arg...)                         \
  do {                                                \
    if (ENV_ENABLE_RAW("KV_IS_DEBUG_ENABLED")) {      \
      GET_CURRENT_TIME();                             \
      fprintf(stdout,                                 \
              "[%s][DBG][KV_CACHE][%s:%d] " fmt "\n", \
              str,                                    \
              FILE_NAME(__FILE__),                    \
              __LINE__,                               \
              ##arg);                                 \
    }                                                 \
  } while (0)
