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
#include "vad.h"

int Vad::getSampleRate() const { return sample_rate_; }

int Vad::getFrameMs() const { return frame_ms_; }

float Vad::getThreshold() const { return threshold_; }

int Vad::getMinSilenceDurationMs() const { return min_silence_duration_ms_; }

int Vad::getSpeechPadMs() const { return speech_pad_ms_; }

const wav::WavReader &Vad::getWavReader() const { return wavReader_; }

const std::vector<int16_t> &Vad::getData() const { return data_; }

const std::vector<float> &Vad::getInputWav() const { return inputWav_; }

int64_t Vad::getWindowSizeSamples() const { return window_size_samples_; }

int Vad::getSrPerMs() const { return sr_per_ms_; }

int Vad::getMinSilenceSamples() const { return min_silence_samples_; }

int Vad::getSpeechPadSamples() const { return speech_pad_samples_; }

std::string Vad::ModelName() const { return "VAD"; }

void Vad::loadAudio(const std::string &wavPath) {
    wavReader_ = wav::WavReader(wavPath);
    data_.reserve(wavReader_.num_samples());
    inputWav_.reserve(wavReader_.num_samples());

    for (int i = 0; i < wavReader_.num_samples(); i++) {
        data_[i] = static_cast<int16_t>(*(wavReader_.data() + i));
    }

    for (int i = 0; i < wavReader_.num_samples(); i++) {
        inputWav_[i] = static_cast<float>(data_[i]) / 32768;
    }
}

bool Vad::Initialize() {
    // initAudioConfig
    sr_per_ms_ = sample_rate_ / 1000;
    min_silence_samples_ = sr_per_ms_ * min_silence_duration_ms_;
    speech_pad_samples_ = sr_per_ms_ * speech_pad_ms_;
    window_size_samples_ = frame_ms_ * sr_per_ms_;

    // initInputConfig
    input_.resize(window_size_samples_);
    input_node_dims_.emplace_back(1);
    input_node_dims_.emplace_back(window_size_samples_);

    _h.resize(size_hc_);
    _c.resize(size_hc_);
    sr_.resize(1);
    sr_[0] = sample_rate_;

    // InitRuntime
    if (!InitRuntime()) {
        fastdeploy::FDERROR << "Failed to initialize fastdeploy backend."
                            << std::endl;
        return false;
    }
    return true;
}

void Vad::setAudioCofig(int sr, int frame_ms, float threshold,
                        int min_silence_duration_ms, int speech_pad_ms) {
    if (initialized) {
        fastdeploy::FDERROR << "setAudioCofig must be called before init"
                            << std::endl;
        throw std::runtime_error("setAudioCofig must be called before init");
    }
    sample_rate_ = sr;
    Vad::frame_ms_ = frame_ms;
    Vad::threshold_ = threshold;
    Vad::min_silence_duration_ms_ = min_silence_duration_ms;
    Vad::speech_pad_ms_ = speech_pad_ms;
}

bool Vad::Preprocess(std::vector<float> audioWindowData) {
    fastdeploy::FDTensor inputTensor, srTensor, hTensor, cTensor;
    inputTensor.SetExternalData(input_node_dims_, fastdeploy::FDDataType::FP32,
                                audioWindowData.data());
    inputTensor.name = "input";
    srTensor.SetExternalData(sr_node_dims_, fastdeploy::FDDataType::INT64,
                             sr_.data());
    srTensor.name = "sr";
    hTensor.SetExternalData(hc_node_dims_, fastdeploy::FDDataType::FP32,
                            _h.data());
    hTensor.name = "h";
    cTensor.SetExternalData(hc_node_dims_, fastdeploy::FDDataType::FP32,
                            _c.data());
    cTensor.name = "c";

    inputTensors_.clear();
    inputTensors_.emplace_back(inputTensor);
    inputTensors_.emplace_back(srTensor);
    inputTensors_.emplace_back(hTensor);
    inputTensors_.emplace_back(cTensor);
    return true;
}

bool Vad::Predict() {
    if (wavReader_.sample_rate() != sample_rate_) {
        fastdeploy::FDINFO << "The sampling rate of the audio file is " << wavReader_.sample_rate() << std::endl;
        fastdeploy::FDINFO << "The set sample rate is " << sample_rate_ << std::endl;
        fastdeploy::FDERROR << "The sampling rate of the audio file is not equal "
                               "to the sampling rate set by the program. "
                            << "Please make it equal. "
                            << "You can modify the audio file sampling rate, "
                            << "or use setAudioCofig to modify the program's "
                               "sampling rate and other configurations."
                            << std::endl;
        throw std::runtime_error(
                "The sampling rate of the audio file is not equal to the sampling rate "
                "set by the program.");
    }
    for (int64_t j = 0; j < wavReader_.num_samples(); j += window_size_samples_) {
        std::vector<float> r{&inputWav_[0] + j,
                             &inputWav_[0] + j + window_size_samples_};
        Preprocess(r);
        if (!Infer(inputTensors_, &outputTensors_)) {
            fastdeploy::FDERROR << "Failed to inference while using model:"
                                << ModelName() << "." << std::endl;
            return false;
        }
        Postprocess();
    }
    return true;
}

bool Vad::Postprocess() {
    // update prob, h, c
    outputProb_ = *(float *)outputTensors_[0].Data();
    auto *hn = static_cast<float *>(outputTensors_[1].MutableData());
    std::memcpy(_h.data(), hn, size_hc_ * sizeof(float));
    auto *cn = static_cast<float *>(outputTensors_[2].MutableData());
    std::memcpy(_c.data(), cn, size_hc_ * sizeof(float));

    // Push forward sample index
    current_sample_ += window_size_samples_;

    if (outputProb_ >= threshold_ && temp_end_) {
        // Reset temp_end_ when > threshold_
        temp_end_ = 0;
    }
    if (outputProb_ < threshold_ && !triggerd_) {
        // 1) Silence
        // printf("{ silence: %.3f s }\n", 1.0 * current_sample_ / sample_rate_);
    }
    if (outputProb_ >= threshold_ - 0.15 && triggerd_) {
        // 2) Speaking
        // printf("{ speaking_2: %.3f s }\n", 1.0 * current_sample_ / sample_rate_);
    }
    if (outputProb_ >= threshold_ && !triggerd_) {
        // 3) Start
        triggerd_ = true;
        speech_start_ = current_sample_ - window_size_samples_ -
                        speech_pad_samples_;  // minus window_size_samples_ to get
        // precise start time point.
        // printf("{ start: %.5f s }\n", 1.0 * speech_start_ / sample_rate_);
        speakStart_.emplace_back(1.0 * speech_start_ / sample_rate_);
    }
    if (outputProb_ < threshold_ - 0.15 && triggerd_) {
        // 4) End
        if (temp_end_ != 0) {
            temp_end_ = current_sample_;
        }
        if (current_sample_ - temp_end_ < min_silence_samples_) {
            // a. silence < min_slience_samples, continue speaking
            // printf("{ speaking_4: %.3f s }\n", 1.0 * current_sample_ / sample_rate_);
            // printf("");
        } else {
            // b. silence >= min_slience_samples, end speaking
            speech_end_ = current_sample_ + speech_pad_samples_;
            temp_end_ = 0;
            triggerd_ = false;
            // printf("{ end: %.5f s }\n", 1.0 * speech_end_ / sample_rate_);
            speakEnd_.emplace_back(1.0 * speech_end_ / sample_rate_);
        }
    }

    return true;
}

std::vector<std::map<std::string, float>> Vad::getResult(
        float removeThreshold, float expandHeadThreshold, float expandTailThreshold,
        float mergeThreshold) {
    float audioLength = 1.0 * wavReader_.num_samples() / sample_rate_;
    if (speakStart_.empty() && speakEnd_.empty()) {
        return {};
    }
    if (speakEnd_.size() != speakStart_.size()) {
        // set the audio length as the last end
        speakEnd_.emplace_back(audioLength);
    }
    // Remove too short segments
    auto startIter = speakStart_.begin();
    auto endIter = speakEnd_.begin();
    while (startIter != speakStart_.end()) {
        if (removeThreshold < audioLength &&
            *endIter - *startIter < removeThreshold) {
            startIter = speakStart_.erase(startIter);
            endIter = speakEnd_.erase(endIter);
        } else {
            startIter++;
            endIter++;
        }
    }
    // Expand to avoid to tight cut.
    startIter = speakStart_.begin();
    endIter = speakEnd_.begin();
    *startIter = std::fmax(0.f, *startIter - expandHeadThreshold);
    *endIter = std::fmin(*endIter + expandTailThreshold, *(startIter + 1));
    endIter = speakEnd_.end() - 1;
    startIter = speakStart_.end() - 1;
    *startIter = fmax(*startIter - expandHeadThreshold, *(endIter - 1));
    *endIter = std::fmin(*endIter + expandTailThreshold, audioLength);
    for (int i = 1; i < speakStart_.size() - 1; ++i) {
        speakStart_[i] = std::fmax(speakStart_[i] - expandHeadThreshold, speakEnd_[i - 1]);
        speakEnd_[i] = std::fmin(speakEnd_[i] + expandTailThreshold, speakStart_[i + 1]);
    }
    // Merge very closed segments
    startIter = speakStart_.begin() + 1;
    endIter = speakEnd_.begin();
    while (startIter != speakStart_.end()) {
        if (*startIter - *endIter < mergeThreshold) {
            startIter = speakStart_.erase(startIter);
            endIter = speakEnd_.erase(endIter);
        } else {
            startIter++;
            endIter++;
        }
    }

    std::vector<std::map<std::string, float>> result;
    for (int i = 0; i < speakStart_.size(); ++i) {
        result.emplace_back(std::map<std::string, float>(
                {{"start", speakStart_[i]}, {"end", speakEnd_[i]}}));
    }
    return result;
}
