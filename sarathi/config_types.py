from enum import Enum
from typing import List, Optional, Tuple

GPULocation = Tuple[Optional[str], int]  # (node_ip, gpu_id)
ResourceMapping = List[GPULocation]
ReplicaResourceMapping = List[ResourceMapping]  # List ResourceMapping for each replica


class SchedulerType(Enum):
    SARATHI = "SARATHI"
    DISAGG_EMULATION = "DISAGG_EMULATION"
    FCFS_DISAGG_EMULATION = "FCFS_DISAGG_EMULATION"
    MLFQ_DISAGG_EMULATION = "MLFQ_DISAGG_EMULATION"
    ROUND_ROBIN_DISAGG_EMULATION = "ROUND_ROBIN_DISAGG_EMULATION"


class RequestGeneratorType(Enum):
    SYNTHETIC = "SYNTHETIC"
    TRACE = "TRACE"


class RequestIntervalGeneratorType(Enum):
    POISSON = "POISSON"
    GAMMA = "GAMMA"
    STATIC = "STATIC"
    TRACE = "TRACE"


class RequestLengthGeneratorType(Enum):
    UNIFORM = "UNIFORM"
    ZIPF = "ZIPF"
    TRACE = "TRACE"
    FIXED = "FIXED"


class AttentionBackend(Enum):
    FLASHINFER = "FLASHINFER"
    NO_OP = "NO_OP"
