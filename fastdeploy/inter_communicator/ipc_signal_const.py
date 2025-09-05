from dataclasses import dataclass


@dataclass
class ModelWeightsStatus:
    NORMAL = 0
    UPDATING = 1
    CLEARING = -1
    CLEARED = -2


@dataclass
class PrefixTreeStatus:
    NORMAL = 0
    UPDATING = 1
    CLEARING = -1
    CLEARED = -2


@dataclass
class KVCacheStatus:
    NORMAL = 0
    UPDATING = 1
    CLEARING = -1
    CLEARED = -2


@dataclass
class ExistTaskStatus:
    EMPTY = 0
    EXIST = 1
    REFUSE = 2
