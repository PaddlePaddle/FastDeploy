import argparse

from fastdeploy_ic.server.launcher import serve

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference load balance controller launcher")
    parser.add_argument("--grpc-port", type=int, default=9000)
    args = parser.parse_args()
    serve(args)