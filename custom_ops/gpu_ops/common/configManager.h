// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <memory>
#include <mutex>
#include <regex>
#include <limits>

class ConfigManager {
public:
    static ConfigManager& get_instance(const std::string& config_path = "fastdeploy_op_configs.json") {
        static ConfigManager instance(config_path);
        return instance;
    }

    std::string get_best_config(const std::string& op_name, const size_t m, const size_t n, const size_t k) {
        initialize();
        std::string mnk_string = op_name + "-" +
                            std::to_string(update_m(m)) + "x" + std::to_string(n) + "x" + std::to_string(k);
        if (configs_.contains(mnk_string)) {
            return configs_.at(mnk_string);
        }
        return "";
    }

    int64_t update_m(const size_t m) {
        size_t new_m = m;
        if (m < 4) {
            return m;
        } else if (m < 16) {
            return  (m + 3) / 4 * 4;
        } else if (m < 64) {
            return (m + 15) / 16 * 16;
        } else if (m < 256) {
            return (m + 31) / 32 * 32;
        } else if (m < 512) {
            return (m + 63) / 64 * 64;
        } else if (m < 1024) {
            return (m + 127) / 128 * 128;
        } else if (m < 8192) {
            return (m + 1023) / 1024 * 1024;
        } else if (m < 32768) {
            return (m + 4095) / 4096 * 4096;
        } else {
            return 32768;
        }
    }

    void update(const std::string& op_name, const size_t m, const size_t n, const size_t k, const std::string& config) {
        initialize();
        std::string mnk_string = op_name + "-" +
                            std::to_string(update_m(m)) + "x" + std::to_string(n) + "x" + std::to_string(k);
        configs_[mnk_string] = config;
    }

    void print() const {
        std::cout << configs_.dump(4) << std::endl; // Pretty print with 4 spaces
    }

    ~ConfigManager() {
        std::ofstream file(config_path_);
        if (file.is_open()) {
            file << configs_.dump(4); // Pretty print with 4 spaces
            file.close();
        }
    }

private:
    void initialize() {
        if (initialized_) return;
        std::ifstream file(config_path_);
        if (file.is_open()) {
            try {
                file >> configs_;
            } catch (const std::exception& e) {
                std::cerr << "Error reading configs from " << config_path_ << " : " << e.what() << std::endl;
                configs_ = nlohmann::json::object(); // Create an empty JSON object
            }
            file.close();
        } else {
            configs_ = nlohmann::json::object(); // Create an empty JSON object
        }
        initialized_ = true;
    }

    ConfigManager(const std::string& config_path) : config_path_(config_path) {}
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;

    nlohmann::json configs_;
    std::string config_path_;
    bool initialized_{false};
};
