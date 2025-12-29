from dataclasses import dataclass
from typing import List


@dataclass(frozen=True, kw_only=True)
class CacheTask:
    task_id: str
    keys: List[str]
    token_ids: List[int]
    gpu_block_ids: List[int]


@dataclass(frozen=True, kw_only=True)
class ReadStorageTask(CacheTask):
    start_read_block_idx: int
    timeout: float = 30.0


@dataclass(frozen=True, kw_only=True)
class WriteStorageTask(CacheTask):
    timeout: float = 30.0
