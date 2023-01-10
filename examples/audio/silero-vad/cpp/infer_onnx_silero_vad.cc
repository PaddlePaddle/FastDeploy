#include <iostream>

#include "Vad.h"


int main(int argc, char* argv[]) {
    std::string model_file = "../silero_vad.onnx";
    std::string audio_file = "../sample.wav";

    Vad vad(model_file);
    // custom config, but must be set before init
    // vad.setAudioCofig(16000, 64, 0.5f, 0, 0);
    vad.init();
    vad.loadAudio(audio_file);
    vad.Predict();
    std::vector<std::map<std::string, float>> result = vad.getResult();
    for (auto & res : result) {
        std::cout << "speak start: " << res["start"] << " s, end: " << res["end"] << " s" << std::endl;
    }
    return 0;
}
