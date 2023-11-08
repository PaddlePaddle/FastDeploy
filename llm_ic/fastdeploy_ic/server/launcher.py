from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from concurrent import futures
import contextlib
import multiprocessing
import socket
import sys
import asyncio

import grpc

import fastdeploy_ic.proto.ic_pb2_grpc as ic_pb2_grpc
from .api import GRPCInferenceServiceServicer

_PROCESS_COUNT = multiprocessing.cpu_count()
_THREAD_CONCURRENCY = _PROCESS_COUNT


async def _run_server(bind_address):
    """Start a server in a subprocess."""
    options = (("grpc.so_reuseport", 1),)
    server = grpc.aio.server(futures.ThreadPoolExecutor(
            max_workers=_THREAD_CONCURRENCY,
        ),
        options=options)
    ic_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(GRPCInferenceServiceServicer(), server)
    server.add_insecure_port(bind_address)
    await server.start()
    await server.wait_for_termination()

def run(bind_address):
    asyncio.run(_run_server(bind_address))

    


@contextlib.contextmanager
def _reserve_port(port):
    """Create a socket for all subprocesses to use."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    if sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 0:
        raise RuntimeError("Failed to set SO_REUSEPORT.")
    sock.bind(("", port))
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


def serve(args):
    with _reserve_port(args.grpc_port) as port:
        bind_address = "localhost:{}".format(port)
        print("Binding to '%s'", bind_address)
        sys.stdout.flush()
        workers = []
        for _ in range(_PROCESS_COUNT):
            # NOTE: It is imperative that the worker subprocesses be forked before
            # any gRPC servers start up. See
            # https://github.com/grpc/grpc/issues/16001 for more details.
            worker = multiprocessing.Process(
                target=run, args=(bind_address,)
            )
            worker.start()
            workers.append(worker)
        for worker in workers:
            worker.join()


    