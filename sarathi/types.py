from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

GPULocation = Tuple[Optional[str], int]  # (node_ip, gpu_id)
ResourceMapping = List[GPULocation]
ReplicaResourceMapping = List[ResourceMapping]  # List ResourceMapping for each replica


class SchedulerType(Enum):
    VLLM = "VLLM"
    ORCA = "ORCA"
    FASTER_TRANSFORMER = "FASTER_TRANSFORMER"
    SARATHI = "SARATHI"
    SIMPLE_CHUNKING = "SIMPLE_CHUNKING"
    ROLLING_PREEMPTION_PROFILING = "ROLLING_PREEMPTION_PROFILING"
    OCCASIONAL_SWAPPING = "OCCASIONAL_SWAPPING"
    FCFS_DISAGG_EMULATION = "FCFS_DISAGG_EMULATION"


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
