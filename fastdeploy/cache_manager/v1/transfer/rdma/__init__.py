"""
RDMA transfer implementation.

RDMA (Remote Direct Memory Access) provides high-performance,
low-latency data transfer for cross-node KV cache movement.
"""

from .connector import RDMAConnector

__all__ = [
    "RDMAConnector",
]
