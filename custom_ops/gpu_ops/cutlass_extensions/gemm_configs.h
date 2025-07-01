/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

namespace cutlass_extensions
{
// Note: The shapes are in the format MxNxK. The K shape of the runtime config MUST match the K shape
//       in the kernel layout details when doing weight only quantization.
enum class CutlassTileConfig
{
    // Signals that we should run heuristics do choose a config
    Undefined,

    // Signals that we should run heuristics do choose a config
    ChooseWithHeuristic,

    // SiMT config
    CtaShape128x128x8_WarpShape64x64x8,

    // TensorCore configs CTA_N = 128, CTA_K = 64
    // Warp configs for M=16
    CtaShape16x128x64_WarpShape16x32x64,
    // Warp configs for M=32
    CtaShape32x128x64_WarpShape32x32x64,

    // Warp configs for M=64
    CtaShape64x128x64_WarpShape32x64x64,
    CtaShape64x64x128_WarpShape32x64x64,
    CtaShape64x128x64_WarpShape64x32x64,

    // Warp configs for M=128
    CtaShape128x64x64_WarpShape64x32x64,
    CtaShape128x128x64_WarpShape64x32x64,
    CtaShape128x128x64_WarpShape64x64x64,
    CtaShape128x128x64_WarpShape128x32x64,
    CtaShape128x256x64_WarpShape64x64x64,

    // Warp configs for M=256
    CtaShape256x128x64_WarpShape64x64x64,

    // TensorCore config CTA_N = 64, CTA_K = 128
    CtaShape128x64x128_WarpShape64x32x128,

    // TensorCore config CTA_N = 256, CTA_K = 64
    CtaShape16x256x64_WarpShape16x64x64,

    // TensorCore config CTA_N = 256, CTA_K = 128
    CtaShape16x256x128_WarpShape16x64x128

};

enum class SplitKStyle
{
    NO_SPLIT_K,
    SPLIT_K_SERIAL,
    STREAM_K, // Sm80+
    // SPLIT_K_PARALLEL // Not supported yet
};

// New enum for SM100 (Blackwell) Tile Configs
// Placeholder values - actual optimal values need research
enum class CutlassTileConfigSM100
{
    // Signals that we should run heuristics do choose a config
    Undefined,

    // Signals that we should run heuristics do choose a config
    ChooseWithHeuristic,

    // Actual SM100 tile configs based on user input (K-tile is 128B)
    CtaShape64x64x128B,
    CtaShape64x128x128B,
    CtaShape64x256x128B,
    CtaShape128x64x128B,
    CtaShape128x128x128B,
    CtaShape128x256x128B,
    CtaShape256x64x128B,
    CtaShape256x128x128B,
    CtaShape256x256x128B
    // Note: The user-provided list for get_candidate_tiles_sm100 also includes
    // CtaShape128x64x128B and CtaShape256x64x128B for specific FP4 grouped gemm cases.
    // These are already covered by the list above if general suffices.
    // If they need distinct enum values, they should be added.
    // For now, keeping the enum concise with unique shapes mentioned for general use.
};


enum class CutlassTileConfigSM90
{
    // Signals that we should run heuristics do choose a config
    Undefined,

    // Signals that we should run heuristics do choose a config
    ChooseWithHeuristic,

    // CTA configs for M=64
    CtaShape64x16x128B,
    CtaShape64x32x128B,
    CtaShape64x64x128B,
    CtaShape64x128x128B,
    CtaShape64x256x128B,

    // CTA configs for M=128
    CtaShape128x16x128B,
    CtaShape128x32x128B,
    CtaShape128x64x128B,
    CtaShape128x128x128B,
    CtaShape128x256x128B,

    // CTA configs for M=128
    CtaShape256x128x128B,
};

enum class MainloopScheduleType
{
    AUTO // Automatically selects between pingpong and cooperative schedules on Hopper. On older architectures, this
         // defaults to the "legacy" main loop schedule.
};

enum class EpilogueScheduleType
{
    AUTO // Automatically chooses an epilogue schedule compatible with the selected main loop schedule for Hopper. For
         // architectures older than hopper, the epilogue is always performed by the same thread block as the main loop.
};

enum class ClusterShape
{
    ClusterShape_1x1x1,
    ClusterShape_2x1x1,
    ClusterShape_1x2x1,
    ClusterShape_2x2x1,
    ClusterShape_1x8x1,
    ClusterShape_8x1x1
};

struct CutlassGemmConfig
{
    enum CandidateConfigTypeParam : int
    {
        NONE = 0,
        WEIGHT_ONLY = 1u << 0,
        SIMT_ONLY = 1u << 1,
        INT8_ONLY = 1u << 2,
        HOPPER = 1u << 3, // SM90
        GROUPED_GEMM = 1u << 4,
        FP8_ONLY = 1u << 5,
        BLACKWELL = 1u << 6, // SM100
        FP4_ONLY = 1u << 7, // For Blackwell FP4/MXFP4 paths
    };

    CutlassTileConfig tile_config = CutlassTileConfig::ChooseWithHeuristic;
    SplitKStyle split_k_style = SplitKStyle::NO_SPLIT_K;
    int split_k_factor = -1;
    int stages = -1;

    // config options for sm90
    CutlassTileConfigSM90 tile_config_sm90 = CutlassTileConfigSM90::ChooseWithHeuristic;
    MainloopScheduleType mainloop_schedule = MainloopScheduleType::AUTO;
    EpilogueScheduleType epilogue_schedule = EpilogueScheduleType::AUTO;
    ClusterShape cluster_shape = ClusterShape::ClusterShape_1x1x1;
    bool is_sm90 = false;

    // config options for sm100 (Blackwell)
    // Assuming SM100 might use similar schedule/cluster types as SM90 for now.
    // These might need to become SM100-specific if Blackwell introduces new concepts.
    CutlassTileConfigSM100 tile_config_sm100 = CutlassTileConfigSM100::ChooseWithHeuristic;
    // MainloopScheduleType mainloop_schedule_sm100 = MainloopScheduleType::AUTO; // Example if SM100 has different types
    // EpilogueScheduleType epilogue_schedule_sm100 = EpilogueScheduleType::AUTO; // Example
    // ClusterShape cluster_shape_sm100 = ClusterShape::ClusterShape_1x1x1;       // Example
    bool is_sm100 = false;


    CutlassGemmConfig() : is_sm90(false), is_sm100(false) {}

    CutlassGemmConfig(CutlassTileConfig tile_config, SplitKStyle split_k_style, int split_k_factor, int stages)
        : tile_config(tile_config)
        , split_k_style(split_k_style)
        , split_k_factor(split_k_factor)
        , stages(stages)
        , is_sm90(false)
        , is_sm100(false)
    {
    }

    // Constructor for SM90
    CutlassGemmConfig(CutlassTileConfigSM90 tile_config_sm90_in, MainloopScheduleType mainloop_schedule_in,
        EpilogueScheduleType epilogue_schedule_in, ClusterShape cluster_shape_in)
        : tile_config_sm90(tile_config_sm90_in)
        , mainloop_schedule(mainloop_schedule_in)
        , epilogue_schedule(epilogue_schedule_in)
        , cluster_shape(cluster_shape_in)
        , is_sm90(true)
        , is_sm100(false)
    {
    }

    // Constructor for SM100 (Blackwell)
    // Using existing MainloopScheduleType, EpilogueScheduleType, ClusterShape for now.
    // These might need to be new SM100-specific types if Blackwell's TMA differs significantly.
    CutlassGemmConfig(CutlassTileConfigSM100 tile_config_sm100_in, MainloopScheduleType mainloop_schedule_in,
                      EpilogueScheduleType epilogue_schedule_in, ClusterShape cluster_shape_in)
        : tile_config_sm100(tile_config_sm100_in)
        , mainloop_schedule(mainloop_schedule_in) // Potentially use mainloop_schedule_sm100 if types diverge
        , epilogue_schedule(epilogue_schedule_in) // Potentially use epilogue_schedule_sm100
        , cluster_shape(cluster_shape_in)         // Potentially use cluster_shape_sm100
        , is_sm90(false) // Explicitly false
        , is_sm100(true)
    {
    }


    std::string toString() const
    {
        std::stringstream tactic;
        tactic << "Cutlass GEMM Tactic";
        if (is_sm100 && tile_config_sm100 != cutlass_extensions::CutlassTileConfigSM100::ChooseWithHeuristic)
        {
            assert(is_sm100 && !is_sm90 && "Invalid cutlass GEMM config: SM100");
            tactic << "\n\tstyle=TMA_SM100" // Indicate SM100 specific TMA if applicable
                   << "\n\ttile shape ID: " << (int) tile_config_sm100
                   << "\n\tcluster shape ID: " << (int) cluster_shape
                   << "\n\tmainloop sched: " << (int) mainloop_schedule
                   << "\n\tepi sched: " << (int) epilogue_schedule;
        }
        else if (is_sm90 && tile_config_sm90 != cutlass_extensions::CutlassTileConfigSM90::ChooseWithHeuristic)
        {
            assert(is_sm90 && !is_sm100 && "Invalid cutlass GEMM config: SM90");
            tactic << "\n\tstyle=TMA_SM90"
                   << "\n\ttile shape ID: " << (int) tile_config_sm90
                   << "\n\tcluster shape ID: " << (int) cluster_shape
                   << "\n\tmainloop sched: " << (int) mainloop_schedule
                   << "\n\tepi sched: " << (int) epilogue_schedule;
        }
        else if (tile_config != cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic)
        {
            assert(!is_sm90 && !is_sm100 && "Invalid cutlass GEMM config: Compatible");
            tactic << "\n\tstyle=compatible"
                   << "\n\ttile shape ID: " << (int) tile_config
                   << "\n\tstages: " << (int) stages
                   << "\n\tsplit_k_style: " << (int) split_k_style
                   << "\n\tsplit k: " << (int) split_k_factor;
        }
        else
        {
            tactic << "\n\tundefined";
        }
        tactic << "\n";
        return tactic.str();
    }

    void fromString(const std::string& str) {
        std::istringstream stream(str);
        std::string line;

        is_sm90 = false; // Reset flags
        is_sm100 = false;

        while (std::getline(stream, line)) {
            if (line.find("style=TMA_SM100") != std::string::npos) {
                is_sm100 = true;
                is_sm90 = false;
                std::getline(stream, line);
                tile_config_sm100 = static_cast<cutlass_extensions::CutlassTileConfigSM100>(std::stoi(line.substr(line.find(':') + 1)));
                std::getline(stream, line);
                cluster_shape = static_cast<cutlass_extensions::ClusterShape>(std::stoi(line.substr(line.find(':') + 1)));
                std::getline(stream, line);
                mainloop_schedule = static_cast<cutlass_extensions::MainloopScheduleType>(std::stoi(line.substr(line.find(':') + 1)));
                std::getline(stream, line);
                epilogue_schedule = static_cast<cutlass_extensions::EpilogueScheduleType>(std::stoi(line.substr(line.find(':') + 1)));
            } else if (line.find("style=TMA_SM90") != std::string::npos) { // Check for SM90 specific first
                is_sm90 = true;
                is_sm100 = false;
                std::getline(stream, line);
                tile_config_sm90 = static_cast<cutlass_extensions::CutlassTileConfigSM90>(std::stoi(line.substr(line.find(':') + 1)));
                std::getline(stream, line);
                cluster_shape = static_cast<cutlass_extensions::ClusterShape>(std::stoi(line.substr(line.find(':') + 1)));
                std::getline(stream, line);
                mainloop_schedule = static_cast<cutlass_extensions::MainloopScheduleType>(std::stoi(line.substr(line.find(':') + 1)));
                std::getline(stream, line);
                epilogue_schedule = static_cast<cutlass_extensions::EpilogueScheduleType>(std::stoi(line.substr(line.find(':') + 1)));
            } else if (line.find("style=compatible") != std::string::npos) {
                is_sm90 = false;
                is_sm100 = false;
                std::getline(stream, line);
                tile_config = static_cast<cutlass_extensions::CutlassTileConfig>(std::stoi(line.substr(line.find(':') + 1)));
                std::getline(stream, line);
                stages = std::stoi(line.substr(line.find(':') + 1));
                std::getline(stream, line);
                split_k_style = static_cast<cutlass_extensions::SplitKStyle>(std::stoi(line.substr(line.find(':') + 1)));
                std::getline(stream, line);
                split_k_factor = std::stoi(line.substr(line.find(':') + 1));
            }
        }
    }
};

inline std::ostream& operator<<(std::ostream& out, CutlassGemmConfig const& config)
{
    // clang-format off
    if (config.is_sm100)
    {
        out << "tile_config_sm100_enum: " << int(config.tile_config_sm100)
            << ", mainloop_schedule_enum: " << int(config.mainloop_schedule) // Assuming same schedule types for now
            << ", epilogue_schedule_enum: " << int(config.epilogue_schedule) // Assuming same schedule types for now
            << ", cluster_shape_enum: " << int(config.cluster_shape);       // Assuming same cluster types for now
    }
    else if (config.is_sm90)
    {
        out << "tile_config_sm90_enum: " << int(config.tile_config_sm90)
            << ", mainloop_schedule_enum: " << int(config.mainloop_schedule)
            << ", epilogue_schedule_enum: " << int(config.epilogue_schedule)
            << ", cluster_shape_enum: " << int(config.cluster_shape);
    }
    else
    {
        out << "tile_config_enum: " << int(config.tile_config)
            << ", split_k_style_enum: " << int(config.split_k_style)
            << ", split_k_factor: " << config.split_k_factor
            << ", stages: " << config.stages;
    }
    // clang-format on
    return out;
}

} // namespace cutlass_extensions
