
#include "fd_streamer.h"

int main(int argc, char* argv[]) {

  auto streamer = fastdeploy::streamer::FDStreamer();
  streamer.Init("streamer_cfg.yml");
  streamer.Run();
  return 0;
}
