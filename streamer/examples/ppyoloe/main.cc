
#include "fd_streamer.h"

int main(int argc, char* argv[]) {

  auto streamer = fastdeploy::streamer::FDStreamer();
  streamer.Init("test.yml");
  return 0;
}
