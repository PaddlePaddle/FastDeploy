/**
 * @file log.cpp
 * @brief Logging module implementation for key-value cache system
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

#include "log.h"
#include <errno.h>
#include <libgen.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include "util.h"

static int pid = -1;
static __thread int tid = -1;
static char hostname[64];
char global_log_last_error[1024] = "";
FILE *global_debug_file = stdout;
FILE *global_error_file = stdout;
static char global_debug_file_name[PATH_MAX + 1] = "";
static char global_err_file_name[PATH_MAX + 1] = "";
int global_debug_level = -1;
pthread_mutex_t global_debug_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t global_log_file_lock = PTHREAD_MUTEX_INITIALIZER;

void log_file_init(FILE **kv_cache_log_file,
                   const char *kv_cache_log_file_env,
                   char *logFileName) {
  int c = 0;
  char *dfn = logFileName;
  while (c < PATH_MAX && kv_cache_log_file_env[c] != '\0') {
    if (kv_cache_log_file_env[c++] != '%') {
      *dfn++ = kv_cache_log_file_env[c - 1];
      continue;
    }
    switch (kv_cache_log_file_env[c++]) {
      case '%':  // Double %
        *dfn++ = '%';
        break;
      case 'h':  // %h = hostname
        dfn += snprintf(dfn, PATH_MAX, "%s", hostname);
        break;
      case 'p':  // %p = pid
        dfn += snprintf(dfn, PATH_MAX, "%d", pid);
        break;
      default:  // Echo everything we don't understand
        *dfn++ = '%';
        *dfn++ = kv_cache_log_file_env[c - 1];
        break;
    }
  }
  *dfn = '\0';
  if (logFileName[0] != '\0') {
    FILE *file = fopen(logFileName, "w");
    if (file != nullptr) {
      setbuf(file, nullptr);  // disable buffering
      *kv_cache_log_file = file;
    }
  }
}

void recreate_log_file(FILE **kv_cache_log_file, char *logFileName) {
  if (logFileName[0] != '\0') {
    pthread_mutex_lock(&global_log_file_lock);
    FILE *file = fopen(
        logFileName,
        "a");  // Use "a" mode to append if file exists, otherwise create it
    // close the previous log file if it exists
    if (*kv_cache_log_file != NULL && *kv_cache_log_file != file) {
      fclose(*kv_cache_log_file);
      *kv_cache_log_file = NULL;
    }
    if (file != NULL) {
      setbuf(file, NULL);  // disable buffering
      *kv_cache_log_file = file;
    }
    pthread_mutex_unlock(&global_log_file_lock);
  }
}

void debug_init() {
  pthread_mutex_lock(&global_debug_lock);
  if (global_debug_level != -1) {
    pthread_mutex_unlock(&global_debug_lock);
    return;
  }

  const char *kv_cache_debug = std::getenv("KV_IS_DEBUG_ENABLED");
  int tempg_kv_cache_debug_level = -1;

  if (kv_cache_debug == NULL) {
    tempg_kv_cache_debug_level = KV_LOG_LEVEL_INFO;
  } else if (strcasecmp(kv_cache_debug, "0") == 0) {
    tempg_kv_cache_debug_level = KV_LOG_LEVEL_INFO;
  } else if (strcasecmp(kv_cache_debug, "1") == 0) {
    tempg_kv_cache_debug_level = KV_LOG_LEVEL_DEBUG;
  } else if (strcasecmp(kv_cache_debug, "2") == 0) {
    tempg_kv_cache_debug_level = KV_LOG_LEVEL_WARN;
  } else if (strcasecmp(kv_cache_debug, "3") == 0) {
    tempg_kv_cache_debug_level = KV_LOG_LEVEL_ERROR;
  } else {
    tempg_kv_cache_debug_level = KV_LOG_LEVEL_INFO;
  }

  gethostname(hostname, 64);
  pid = getpid();

  const char *g_kv_cache_debug_fileEnv =
      KVCacheConfig::getInstance().get_debug_file_path();
  if (tempg_kv_cache_debug_level >= KV_LOG_LEVEL_INFO &&
      g_kv_cache_debug_fileEnv != NULL) {
    log_file_init(
        &global_debug_file, g_kv_cache_debug_fileEnv, global_debug_file_name);
  }

  const char *g_kv_cache_error_fileEnv =
      KVCacheConfig::getInstance().get_error_file_path();
  if (tempg_kv_cache_debug_level >= KV_LOG_LEVEL_INFO &&
      g_kv_cache_error_fileEnv != NULL) {
    log_file_init(
        &global_error_file, g_kv_cache_error_fileEnv, global_err_file_name);
    char buffer[1024];
    size_t len = 0;
    char timeBuffer[80];  // Buffer to hold the formatted time
    std::time_t absoluteTime =
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::strftime(timeBuffer,
                  sizeof(timeBuffer),
                  "%Y-%m-%d %H:%M:%S",
                  std::localtime(&absoluteTime));
    len = snprintf(buffer, sizeof(buffer), "%s KV_CACHE START ", timeBuffer);
    buffer[len++] = '\n';
    if (global_error_file != NULL) {
      fwrite(buffer, 1, len, global_error_file);
    }
  }
  __atomic_store_n(
      &global_debug_level, tempg_kv_cache_debug_level, __ATOMIC_RELEASE);
  pthread_mutex_unlock(&global_debug_lock);
}

/* Common logging function used by the INFO, DEBUG and WARN macros
 * Also exported to the dynamically loadable Net transport modules so
 * they can share the debugging mechanisms and output files
 */
void debug_log(KVLogLevel level,
               bool enable_to_terminal,
               const char *filefunc,
               int line,
               const char *fmt,
               ...) {
  if (__atomic_load_n(&global_debug_level, __ATOMIC_ACQUIRE) == -1) {
    debug_init();
  }

  // Save the last error (WARN) as a human readable string
  if (level == KV_LOG_LEVEL_WARN) {
    pthread_mutex_lock(&global_debug_lock);
    va_list vargs;
    va_start(vargs, fmt);
    (void)vsnprintf(
        global_log_last_error, sizeof(global_log_last_error), fmt, vargs);
    va_end(vargs);
    pthread_mutex_unlock(&global_debug_lock);
  }

  if (tid == -1) {
    tid = syscall(SYS_gettid);
  }

  char buffer[1024];
  size_t len = 0;
  // Convert timestamp to absolute time and directly use it in the snprintf
  // function
  std::time_t absoluteTime =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  char timeBuffer[80];  // Buffer to hold the formatted time
  std::strftime(timeBuffer,
                sizeof(timeBuffer),
                "%Y-%m-%d %H:%M:%S",
                std::localtime(&absoluteTime));

  if (level == KV_LOG_LEVEL_WARN) {
    len = snprintf(buffer,
                   sizeof(buffer),
                   "\n%s %s:%d:%d  %s:%d KV_CACHE WARN ",
                   timeBuffer,
                   hostname,
                   pid,
                   tid,
                   filefunc,
                   line);
  } else if (level == KV_LOG_LEVEL_INFO) {
    len = snprintf(buffer,
                   sizeof(buffer),
                   "%s %s:%d:%d KV_CACHE INFO ",
                   timeBuffer,
                   hostname,
                   pid,
                   tid);
  } else if (level == KV_LOG_LEVEL_DEBUG) {
    len = snprintf(buffer,
                   sizeof(buffer),
                   "%s %s:%d:%d KV_CACHE DEBUG ",
                   timeBuffer,
                   hostname,
                   pid,
                   tid);
  } else if (level == KV_LOG_LEVEL_ERROR) {
    len = snprintf(buffer,
                   sizeof(buffer),
                   "%s %s:%d:%d KV_CACHE ERROR ",
                   timeBuffer,
                   hostname,
                   pid,
                   tid);
  } else {
    len = snprintf(buffer,
                   sizeof(buffer),
                   "%s %s:%d:%d KV_CACHE ",
                   timeBuffer,
                   hostname,
                   pid,
                   tid);
  }

  if (len) {
    va_list vargs;
    va_start(vargs, fmt);
    len += vsnprintf(buffer + len, sizeof(buffer) - len, fmt, vargs);
    va_end(vargs);
    // vsnprintf may return len > sizeof(buffer) in the case of a truncated
    // output. Rewind len so that we can replace the final \0 by \n
    if (len > sizeof(buffer)) {
      len = sizeof(buffer) - 1;
    }
    buffer[len++] = '\n';
    if (access(global_debug_file_name, F_OK) != 0) {
      recreate_log_file(&global_debug_file, global_debug_file_name);
    }
    if (enable_to_terminal) {
      fwrite(buffer, 1, len, global_debug_file);
    }
    if (level == KV_LOG_LEVEL_WARN && global_error_file != stdout) {
      if (access(global_err_file_name, F_OK) != 0) {
        recreate_log_file(&global_error_file, global_err_file_name);
      }
      if (global_error_file != NULL) {
        fwrite(buffer, 1, len, global_error_file);
      }
    }
  }
}
