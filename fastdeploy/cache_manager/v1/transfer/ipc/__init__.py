"""
IPC transfer implementation.

IPC (Inter-Process Communication) provides data transfer for
cross-process KV cache movement on the same node.
"""

from .connector import IPCConnector

__all__ = [
    "IPCConnector",
]
