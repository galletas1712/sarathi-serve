import enum
from typing import Union


class SequenceStatus(enum.Enum):
    """Status of a sequence."""

    WAITING = enum.auto()
    RUNNING = enum.auto()
    PAUSED = enum.auto()
    SWAPPING_IN = enum.auto()
    SWAPPING_OUT = enum.auto()
    SWAPPED_OUT = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    @staticmethod
    def is_finished(status: "SequenceStatus") -> bool:
        return status in [
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH_CAPPED,
            SequenceStatus.FINISHED_IGNORED,
        ]

    @staticmethod
    def is_executing(status: "SequenceStatus") -> bool:
        return status in [
            SequenceStatus.RUNNING,
            SequenceStatus.PAUSED,
        ]

    @staticmethod
    def is_waiting(status: "SequenceStatus") -> bool:
        return status == SequenceStatus.WAITING
    
    @staticmethod
    def is_swapping_in(status: "SequenceStatus") -> bool:
        return status == SequenceStatus.SWAPPING_IN
    
    @staticmethod
    def is_swapping_out(status: "SequenceStatus") -> bool:
        return status == SequenceStatus.SWAPPING_OUT
    
    @staticmethod
    def is_swapped_out(status: "SequenceStatus") -> bool:
        return status == SequenceStatus.SWAPPED_OUT

    @staticmethod
    def is_paused(status: "SequenceStatus") -> bool:
        return status == SequenceStatus.PAUSED

    @staticmethod
    def is_running(status: "SequenceStatus") -> bool:
        return status == SequenceStatus.RUNNING

    @staticmethod
    def get_finished_reason(status: "SequenceStatus") -> Union[str, None]:
        if status == SequenceStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif status == SequenceStatus.FINISHED_LENGTH_CAPPED:
            finish_reason = "length"
        elif status == SequenceStatus.FINISHED_IGNORED:
            finish_reason = "length"
        else:
            finish_reason = None
        return finish_reason
