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

int Vad::getSampleRate() const { return sample_rate; }

int Vad::getFrameMs() const { return frame_ms; }

float Vad::getThreshold() const { return threshold; }

int Vad::getMinSilenceDurationMs() const { return min_silence_duration_ms; }

int Vad::getSpeechPadMs() const { return speech_pad_ms; }

const wav::WavReader &Vad::getWavReader() const { return wavReader; }

const std::vector<int16_t> &Vad::getData() const { return data; }

const std::vector<float> &Vad::getInputWav() const { return inputWav; }

int64_t Vad::getWindowSizeSamples() const { return window_size_samples; }

int Vad::getSrPerMs() const { return sr_per_ms; }

int Vad::getMinSilenceSamples() const { return min_silence_samples; }

int Vad::getSpeechPadSamples() const { return speech_pad_samples; }

std::string Vad::ModelName() const { return "VAD"; }

void Vad::loadAudio(const std::string &wavPath) {
  wavReader = wav::WavReader(wavPath);
  data.reserve(wavReader.num_samples());
  inputWav.reserve(wavReader.num_samples());

  for (int i = 0; i < wavReader.num_samples(); i++) {
    data[i] = static_cast<int16_t>(*(wavReader.data() + i));
  }

  for (int i = 0; i < wavReader.num_samples(); i++) {
    inputWav[i] = static_cast<float>(data[i]) / 32768;
  }
}

bool Vad::Initialize() {
  // initAudioConfig
  sr_per_ms = sample_rate / 1000;
  min_silence_samples = sr_per_ms * min_silence_duration_ms;
  speech_pad_samples = sr_per_ms * speech_pad_ms;
  window_size_samples = frame_ms * sr_per_ms;

  // initInputConfig
  input.resize(window_size_samples);
  input_node_dims.emplace_back(1);
  input_node_dims.emplace_back(window_size_samples);

  _h.resize(size_hc);
  _c.resize(size_hc);
  sr.resize(1);
  sr[0] = sample_rate;

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
  sample_rate = sr;
  Vad::frame_ms = frame_ms;
  Vad::threshold = threshold;
  Vad::min_silence_duration_ms = min_silence_duration_ms;
  Vad::speech_pad_ms = speech_pad_ms;
}

bool Vad::Preprocess(std::vector<float> audioWindowData) {
  fastdeploy::FDTensor inputTensor, srTensor, hTensor, cTensor;
  inputTensor.SetExternalData(input_node_dims, fastdeploy::FDDataType::FP32,
                              audioWindowData.data());
  inputTensor.name = "input";
  srTensor.SetExternalData(sr_node_dims, fastdeploy::FDDataType::INT64,
                           sr.data());
  srTensor.name = "sr";
  hTensor.SetExternalData(hc_node_dims, fastdeploy::FDDataType::FP32,
                          _h.data());
  hTensor.name = "h";
  cTensor.SetExternalData(hc_node_dims, fastdeploy::FDDataType::FP32,
                          _c.data());
  cTensor.name = "c";

  inputTensors.clear();
  inputTensors.emplace_back(inputTensor);
  inputTensors.emplace_back(srTensor);
  inputTensors.emplace_back(hTensor);
  inputTensors.emplace_back(cTensor);
  return true;
}

bool Vad::Predict() {
  if (wavReader.sample_rate() != sample_rate) {
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
  for (int64_t j = 0; j < wavReader.num_samples(); j += window_size_samples) {
    std::vector<float> r{&inputWav[0] + j,
                         &inputWav[0] + j + window_size_samples};
    Preprocess(r);
    if (!Infer(inputTensors, &outputTensors)) {
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
  outputProb = *(float *)outputTensors[0].Data();
  auto *hn = static_cast<float *>(outputTensors[1].MutableData());
  std::memcpy(_h.data(), hn, size_hc * sizeof(float));
  auto *cn = static_cast<float *>(outputTensors[2].MutableData());
  std::memcpy(_c.data(), cn, size_hc * sizeof(float));

  // Push forward sample index
  current_sample += window_size_samples;

  if (outputProb >= threshold && temp_end) {
    // Reset temp_end when > threshold
    temp_end = 0;
  }
  if (outputProb < threshold && !triggerd) {
    // 1) Silence
    // printf("{ silence: %.3f s }\n", 1.0 * current_sample / sample_rate);
  }
  if (outputProb >= threshold - 0.15 && triggerd) {
    // 2) Speaking
    // printf("{ speaking_2: %.3f s }\n", 1.0 * current_sample / sample_rate);
  }
  if (outputProb >= threshold && !triggerd) {
    // 3) Start
    triggerd = true;
    speech_start = current_sample - window_size_samples -
                   speech_pad_samples;  // minus window_size_samples to get
                                        // precise start time point.
    // printf("{ start: %.5f s }\n", 1.0 * speech_start / sample_rate);
    speakStart.emplace_back(1.0 * speech_start / sample_rate);
  }
  if (outputProb < threshold - 0.15 && triggerd) {
    // 4) End
    if (temp_end != 0) {
      temp_end = current_sample;
    }
    if (current_sample - temp_end < min_silence_samples) {
      // a. silence < min_slience_samples, continue speaking
      // printf("{ speaking_4: %.3f s }\n", 1.0 * current_sample / sample_rate);
      // printf("");
    } else {
      // b. silence >= min_slience_samples, end speaking
      speech_end = current_sample + speech_pad_samples;
      temp_end = 0;
      triggerd = false;
      // printf("{ end: %.5f s }\n", 1.0 * speech_end / sample_rate);
      speakEnd.emplace_back(1.0 * speech_end / sample_rate);
    }
  }

  return true;
}

std::vector<std::map<std::string, float>> Vad::getResult(
    float removeThreshold, float expandHeadThreshold, float expandTailThreshold,
    float mergeThreshold) {
  float audioLength = 1.0 * wavReader.num_samples() / sample_rate;
  if (speakStart.empty() && speakEnd.empty()) {
    return {};
  }
  if (speakEnd.size() != speakStart.size()) {
    // set the audio length as the last end
    speakEnd.emplace_back(audioLength);
  }
  // Remove too short segments
  auto startIter = speakStart.begin();
  auto endIter = speakEnd.begin();
  while (startIter != speakStart.end()) {
    if (removeThreshold < audioLength &&
        *endIter - *startIter < removeThreshold) {
      startIter = speakStart.erase(startIter);
      endIter = speakEnd.erase(endIter);
    } else {
      startIter++;
      endIter++;
    }
  }
  // Expand to avoid to tight cut.
  startIter = speakStart.begin();
  endIter = speakEnd.begin();
  *startIter = fmax(0.f, *startIter - expandHeadThreshold);
  *endIter = fmin(*endIter + expandTailThreshold, *(startIter + 1));
  endIter = speakEnd.end() - 1;
  startIter = speakStart.end() - 1;
  *startIter = fmax(*startIter - expandHeadThreshold, *(endIter - 1));
  *endIter = fmin(*endIter + expandTailThreshold, audioLength);
  for (int i = 1; i < speakStart.size() - 1; ++i) {
    speakStart[i] = fmax(speakStart[i] - expandHeadThreshold, speakEnd[i - 1]);
    speakEnd[i] = fmin(speakEnd[i] + expandTailThreshold, speakStart[i + 1]);
  }
  // Merge very closed segments
  startIter = speakStart.begin() + 1;
  endIter = speakEnd.begin();
  while (startIter != speakStart.end()) {
    if (*startIter - *endIter < mergeThreshold) {
      startIter = speakStart.erase(startIter);
      endIter = speakEnd.erase(endIter);
    } else {
      startIter++;
      endIter++;
    }
  }

  std::vector<std::map<std::string, float>> result;
  for (int i = 0; i < speakStart.size(); ++i) {
    result.emplace_back(std::map<std::string, float>(
        {{"start", speakStart[i]}, {"end", speakEnd[i]}}));
  }
  return result;
}
