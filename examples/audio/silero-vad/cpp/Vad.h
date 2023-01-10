// Copyright (c) 2023 Chen Qianhe Authors. All Rights Reserved.
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
#ifndef SILERO_VAD_VAD_H
#define SILERO_VAD_VAD_H

#include <cmath>
#include <vector>

#include "fastdeploy/runtime.h"
#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/vision/utils/utils.h"
#include "wav.h"

class Vad:public fastdeploy::FastDeployModel{

public:
    std::string ModelName() const override;

    Vad(const std::string& model_file,
        const fastdeploy::RuntimeOption& custom_option = fastdeploy::RuntimeOption()){

        valid_cpu_backends = {fastdeploy::Backend::ORT, fastdeploy::Backend::OPENVINO};
        valid_gpu_backends = {fastdeploy::Backend::ORT, fastdeploy::Backend::TRT};

        runtime_option = custom_option;
        runtime_option.model_format = fastdeploy::ModelFormat::ONNX;
        runtime_option.model_file = model_file;
        runtime_option.params_file = "";
    };

    void init() {
        initialized = Initialize();
    }

    void setAudioCofig(int sr, int frame_ms, float threshold, int min_silence_duration_ms, int speech_pad_ms);

    void loadAudio(const std::string& wavPath);

    bool Predict();

    std::vector<std::map<std::string, float>> getResult(
            float removeThreshold = 1.6, float expandHeadThreshold = 0.32, float expandTailThreshold = 0,
            float mergeThreshold = 0.3);

private:
    bool Initialize();

    bool Preprocess(std::vector<float> audioWindowData);

    bool Postprocess();

private:
    // model
    std::vector<fastdeploy::FDTensor> inputTensors;
    std::vector<fastdeploy::FDTensor> outputTensors;
    // model states
    bool triggerd = false;
    unsigned int speech_start = 0;
    unsigned int speech_end = 0;
    unsigned int temp_end = 0;
    unsigned int current_sample = 0;
    // MAX 4294967295 samples / 8sample per ms / 1000 / 60 = 8947 minutes
    float outputProb;

/* ============================================================================= */

    // input wav data
    wav::WavReader wavReader;
    std::vector<int16_t> data;
    std::vector<float> inputWav;

/* ============================================================================= */

    // audio config
    int sample_rate = 16000;
    int frame_ms = 64;
    float threshold = 0.5f;
    int min_silence_duration_ms = 0;
    int speech_pad_ms = 0;

    int64_t window_size_samples;
    // Assign when init, support 256 512 768 for 8k; 512 1024 1536 for 16k.
    int sr_per_ms;  // Assign when init, support 8 or 16
    int min_silence_samples; // sr_per_ms * #ms
    int speech_pad_samples; // usually a

/* ============================================================================= */

    std::vector<float> input;
    std::vector<int64_t> sr;
    unsigned int size_hc = 2 * 1 * 64; // It's FIXED.
    std::vector<float> _h;
    std::vector<float> _c;

    std::vector<int64_t> input_node_dims;
    const std::vector<int64_t> sr_node_dims = {1};
    const std::vector<int64_t> hc_node_dims = {2, 1, 64};

/* ============================================================================= */

    std::vector<float> speakStart;
    std::vector<float> speakEnd;

public:
    int getSampleRate() const;

    int getFrameMs() const;

    float getThreshold() const;

    int getMinSilenceDurationMs() const;

    int getSpeechPadMs() const;

    const wav::WavReader &getWavReader() const;

    const std::vector<int16_t> &getData() const;

    const std::vector<float> &getInputWav() const;

    int64_t getWindowSizeSamples() const;

    int getSrPerMs() const;

    int getMinSilenceSamples() const;

    int getSpeechPadSamples() const;
};


#endif //SILERO_VAD_VAD_H
