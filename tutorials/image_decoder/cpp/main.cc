#include "fastdeploy/vision/common/image_decoder/image_decoder.h"

namespace fdvis = fastdeploy::vision;
namespace fd = fastdeploy;

void OpenCVImageDecode(const std::string& img_name) {
  fdvis::FDMat mat;
  auto img_decoder = new fdvis::ImageDecoder();
  img_decoder->Decode(img_name, &mat);
  mat.PrintInfo("");
  delete img_decoder;
}

void NvJpegImageDecode(const std::string& img_name) {
  std::vector<fdvis::FDMat> mats(1);
  std::vector<fastdeploy::FDTensor> caches(1);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  // For nvJPEG decoder, we need set stream and output cache for the FDMat
  for (size_t i = 0; i < mats.size(); i++) {
    mats[i].output_cache = &caches[i];
    mats[i].SetStream(stream);
  }
  auto img_decoder = new fdvis::ImageDecoder(fdvis::ImageDecoderLib::NVJPEG);

  // This is batch decode API, for single image decode API,
  // please refer to OpenCVImageDecode()
  img_decoder->BatchDecode({img_name}, &mats);

  for (size_t i = 0; i < mats.size(); i++) {
    std::cout << "Mat type: " << mats[i].mat_type << ", "
              << "DataType=" << mats[i].Type() << ", "
              << "Channel=" << mats[i].Channels() << ", "
              << "Height=" << mats[i].Height() << ", "
              << "Width=" << mats[i].Width() << std::endl;
  }

  cudaStreamDestroy(stream);
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "Usage: image_decoder path/to/image run_option, "
                 "e.g ./image_decoder ./test.jpeg 0"
              << std::endl;
    std::cout << "Run_option 0: OpenCV; 1: nvJPEG " << std::endl;
    return -1;
  }

  if (std::atoi(argv[2]) == 0) {
    OpenCVImageDecode(argv[1]);
  } else if (std::atoi(argv[2]) == 1) {
    NvJpegImageDecode(argv[1]);
  }
  return 0;
}