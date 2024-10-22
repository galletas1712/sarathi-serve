import time

from sarathi.core.datatypes.sequence_status import SequenceStatus


class SequenceState:

    def __init__(self, id: str, arrived_at: float, num_prompt_tokens: int):
        self._id = id
        self._status = SequenceStatus.WAITING

    def _handle_transitions_from_waiting_status(
        self, current_time: float, status: SequenceStatus, **kwargs
    ) -> None:
        # NOTE: Transitions from running are more intricate
        if status != SequenceStatus.RUNNING and status != SequenceStatus.FINISHED_IGNORED:
            raise ValueError(
                f"Invalid state transition from {self._status} to {status} for request {self._id}."
            )

    def _handle_transitions_from_running_status(
        self, current_time: float, status: SequenceStatus, **kwargs
    ) -> None:
        if status != SequenceStatus.PAUSED and status != SequenceStatus.WAITING:
            raise ValueError(
                f"Invalid state transition from {self._status} to {status} for request {self._id}."
            )

    def _handle_transitions_from_paused_status(
        self, current_time: float, status: SequenceStatus, **kwargs
    ) -> None:
        if not (
            status == SequenceStatus.FINISHED_STOPPED
            or status == SequenceStatus.FINISHED_LENGTH_CAPPED
        ) and status != SequenceStatus.RUNNING and status != SequenceStatus.WAITING and status != SequenceStatus.SWAPPING_OUT:
            raise ValueError(
                f"Invalid state transition from {self._status} to {status} for request {self._id}."
            )

    def _handle_transitions_from_swapping_out_status(
        self, current_time: float, status: SequenceStatus, **kwargs
    ) -> None:
        if status != SequenceStatus.SWAPPED_OUT:
            raise ValueError(
                f"Invalid state transition from {self._status} to {status} for request {self._id}."
            )

    def _handle_transitions_from_swapped_status(
        self, current_time: float, status: SequenceStatus, **kwargs
    ) -> None:
        if status != SequenceStatus.SWAPPING_IN:
            raise ValueError(
                f"Invalid state transition from {self._status} to {status} for request {self._id}."
            )
    
    def _handle_transitions_from_swapping_in_status(
        self, current_time: float, status: SequenceStatus, **kwargs
    ) -> None:
        if status != SequenceStatus.PAUSED:
            raise ValueError(
                f"Invalid state transition from {self._status} to {status} for request {self._id}."
            )
    
    def set_status(self, status: SequenceStatus, **kwargs) -> None:
        current_time = time.monotonic()

        if self._status == SequenceStatus.WAITING:
            self._handle_transitions_from_waiting_status(current_time, status, **kwargs)
        elif self._status == SequenceStatus.RUNNING:
            self._handle_transitions_from_running_status(current_time, status, **kwargs)
        elif self._status == SequenceStatus.PAUSED:
            self._handle_transitions_from_paused_status(current_time, status, **kwargs)
        elif self._status == SequenceStatus.SWAPPED_OUT:
            self._handle_transitions_from_swapped_status(current_time, status, **kwargs)
        elif self._status == SequenceStatus.SWAPPING_OUT:
            self._handle_transitions_from_swapping_out_status(current_time, status, **kwargs)
        elif self._status == SequenceStatus.SWAPPING_IN:
            self._handle_transitions_from_swapping_in_status(current_time, status, **kwargs)
        else:
            raise ValueError(
                f"Invalid state transition from {self._status} to {status} for request {self._id}."
            )

        self._status = status