import enum
from typing import Union


class SequenceStatus(enum.Enum):
    """Status of a sequence."""

    WAITING = enum.auto()
    RUNNING = enum.auto()
    PAUSED = enum.auto()
    SWAPPING_IN = enum.auto()
    SWAPPED_OUT = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    @staticmethod
    def check_transition(start_status: "SequenceStatus", end_status: "SequenceStatus") -> bool:
        ALLOWED_TRANSITIONS = {
            SequenceStatus.WAITING: [
                SequenceStatus.RUNNING,
                SequenceStatus.FINISHED_IGNORED,
            ],
            SequenceStatus.RUNNING: [
                SequenceStatus.PAUSED,
                SequenceStatus.WAITING,
            ],
            SequenceStatus.PAUSED: [
                SequenceStatus.FINISHED_STOPPED,
                SequenceStatus.FINISHED_LENGTH_CAPPED,
                SequenceStatus.RUNNING,
                SequenceStatus.WAITING,
                SequenceStatus.SWAPPED_OUT,
            ],
            SequenceStatus.SWAPPED_OUT: [
                SequenceStatus.SWAPPED_OUT,  # NOTE: We can keep on evicting more tokens from a request
                SequenceStatus.SWAPPING_IN,
            ],
            SequenceStatus.SWAPPING_IN: [
                SequenceStatus.PAUSED,
            ],
            SequenceStatus.FINISHED_IGNORED: [],
            SequenceStatus.FINISHED_STOPPED: [],
            SequenceStatus.FINISHED_LENGTH_CAPPED: []
        }
        
        assert end_status in ALLOWED_TRANSITIONS[start_status], f"Invalid state transition from {start_status} to {end_status}"

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

