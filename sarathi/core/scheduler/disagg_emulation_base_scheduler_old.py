import enum
import time
from typing import List

from sarathi.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SarathiSchedulerConfig,
)
from sarathi.core.block_space_manager.sarathi_block_space_manager import (
    SarathiBlockSpaceManager,
)
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from sarathi.core.scheduler.base_scheduler import BaseScheduler
from sarathi.logger import init_logger

logger = init_logger(__name__)


class DisaggEmulationSchedulerStatus(enum.Enum):
    START = enum.auto()
    PREFILL = enum.auto()
    DECODE = enum.auto()
    DECODE_TO_PREFILL_SWITCHING = enum.auto()
    PREFILL_TO_DECODE_SWITCHING = enum.auto()

    @staticmethod
    def is_start(status: "DisaggEmulationSchedulerStatus") -> bool:
        return status == DisaggEmulationSchedulerStatus.START
    
    @staticmethod
    def is_prefill(status: "DisaggEmulationSchedulerStatus") -> bool:
        return status == DisaggEmulationSchedulerStatus.PREFILL
    
    @staticmethod
    def is_decode(status: "DisaggEmulationSchedulerStatus") -> bool:
        return status == DisaggEmulationSchedulerStatus.DECODE
    
    @staticmethod
    def is_switching_decode_to_prefill(status: "DisaggEmulationSchedulerStatus") -> bool:
        return status == DisaggEmulationSchedulerStatus.DECODE_TO_PREFILL_SWITCHING
    
    @staticmethod
    def is_switching_prefill_to_decode(status: "DisaggEmulationSchedulerStatus") -> bool:
        return status == DisaggEmulationSchedulerStatus.PREFILL_TO_DECODE_SWITCHING


class DisaggEmulationBaseScheduler(BaseScheduler):

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: SarathiSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        super().__init__(model_config, scheduler_config, cache_config, parallel_config)

        self.fsm_state = None
        self.switch_decode_to_prefill_seq_ids: List[str] = []
        self.switch_prefill_to_decode_seq_ids: List[str] = []
        self.switch_swapped_out_seq_ids: List[str] = []

    def get_block_space_manager_class(self):
        return SarathiBlockSpaceManager

    def _get_seq_next_num_prefill_tokens(
        self, seq: Sequence, num_batched_tokens: int
    ) -> int:
        assert not seq.is_finished()
        next_num_tokens = min(
            seq.get_prompt_len() - seq.get_num_prompt_tokens_stage_processed(),
            self.chunk_size - num_batched_tokens,
        )

        return next_num_tokens
    
    def _is_start(self) -> bool:
        return DisaggEmulationSchedulerStatus.is_start(self.fsm_state)
    
    def _is_prefill(self) -> bool:
        return DisaggEmulationSchedulerStatus.is_prefill(self.fsm_state)
    
    def _is_decode(self) -> bool:
        return DisaggEmulationSchedulerStatus.is_decode(self.fsm_state)

    def _is_switching_decode_to_prefill(self) -> bool:
        return DisaggEmulationSchedulerStatus.is_switching_decode_to_prefill(self.fsm_state)
    
    def _is_switching_prefill_to_decode(self) -> bool:
        return DisaggEmulationSchedulerStatus.is_switching_prefill_to_decode(self.fsm_state)
    
    def _finish_swap_in(self, seq: Sequence) -> None:
        super()._finish_swap_in(seq)
        # Updates to emulation-specific FSM state
        if seq.seq_id in self.switch_prefill_to_decode_seq_ids:
            self.switch_prefill_to_decode_seq_ids.remove(seq.seq_id)
    
    def _finish_swap_out(self, seq: Sequence) -> None:
        super()._finish_swap_out(seq)
        # Updates to emulation-specific FSM state
        if seq.seq_id in self.switch_decode_to_prefill_seq_ids:
            self.switch_decode_to_prefill_seq_ids.remove(seq.seq_id)
            self.switch_swapped_out_seq_ids.append(seq.seq_id)
    
    def _schedule(self) -> SchedulerOutputs:
        # Fix the current time.
        now = time.monotonic()

        # Get running prefills and running decodes
        running_prefills = []
        running_decodes = []
        for seq in self.running:
            assert seq.is_paused()

            if not seq.prompt_stage_processing_finished:
                running_prefills.append(seq)
            else:
                running_decodes.append(seq)
        assert not running_prefills or not running_decodes

        # NOTE: We're clearing the running list here
        self.running = []

        # Update FSM state
        if self.switch_decode_to_prefill_seq_ids:
            self.fsm_state = DisaggEmulationSchedulerStatus.DECODE_TO_PREFILL_SWITCHING
        elif self.switch_prefill_to_decode_seq_ids:
            self.fsm_state = DisaggEmulationSchedulerStatus.PREFILL_TO_DECODE_SWITCHING
        else:
            if running_prefills or can_schedule_waiting_prefills():
                if running_decodes:
                    # This case means we want to schedule new prefills but there's decodes running
                    self.fsm_state = DisaggEmulationSchedulerStatus.DECODE_TO_PREFILL_SWITCHING
                else:
                    self.fsm_state = DisaggEmulationSchedulerStatus.PREFILL
            else:
                if self.swapped_out:
                    self.fsm_state = DisaggEmulationSchedulerStatus.PREFILL_TO_DECODE_SWITCHING
                else:
                    self.fsm_state = DisaggEmulationSchedulerStatus.DECODE
            
        running: List[Sequence] = []
        ignored_seq_ids: List[str] = []
        preempted_seq_ids: List[str] = []
        begin_swap_in_seq_ids: List[str] = []
        begin_swap_out_seq_ids: List[str] = []
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata] = []

        num_batched_tokens: int = 0

        if self._is_switching_decode_to_prefill():
            # We updated the FSM state but still need to queue up more swap outs
            for seq in running_decodes:
                self._begin_swap_out(seq)
                begin_swap_out_seq_ids.append(seq.seq_id)
                self.switch_decode_to_prefill_seq_ids.append(seq.seq_id)
        elif self._is_switching_prefill_to_decode():
            # We updated the FSM state but still need to queue up more swap ins
            for seq in self.switch_swapped_out_seq_ids:
                self._begin_swap_in(seq)
                begin_swap_in_seq_ids.append(seq.seq_id)
                self.switch_prefill_to_decode_seq_ids.append(seq.seq_id)
            self.switch_swapped_out_seq_ids = []
            
        elif self._is_prefill():
            # Schedule currently running prefills
            for seq in running_prefills:
                assert not seq.prompt_stage_processing_finished

                next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                    seq, num_batched_tokens
                )

                # as long as the request could fit in the batch previously
                # it should be able to fit in the batch now
                # so in non-pipeline case this condition should always be false
                # however, in pipeline case, the grouping of requests can change
                # between different microbatches, so this is not guaranteed to be always true
                if next_num_prefill_tokens == 0:
                    running.append(seq)
                    continue

                num_batched_tokens += next_num_prefill_tokens
                scheduled_seq_metadata_list.append(
                    SequenceScheduleMetadata.from_sequence(
                        seq, prompt_chunk_len=next_num_prefill_tokens
                    )
                )
                running.append(seq)

            # Schedule new prefills
            while self.waiting:
                seq = self.waiting[0]

                # This is required to handle benchmarking where we set request arrival time ahead of time
                if seq.arrival_time > now:
                    break

                if not self._check_request_prompt_length(seq):
                    ignored_seq_ids.append(seq.seq_id)
                    continue

                # If the sequence group cannot be allocated, stop.
                if not self.block_manager.can_allocate(seq):
                    # this is different from vllm scheduler
                    # even if we cannot allocate this sequence group
                    # there might be other sequence groups that can be allocated
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                if len(running) >= self.scheduler_config.max_num_seqs:
                    break

                # check if we can fit the prefill in the batch
                next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                    seq, num_batched_tokens
                )

                if next_num_prefill_tokens == 0:
                    break

                seq = self.waiting.pop(0)
                self._allocate(seq)
                num_batched_tokens += next_num_prefill_tokens
                scheduled_seq_metadata_list.append(
                    SequenceScheduleMetadata.from_sequence(
                        seq, prompt_chunk_len=next_num_prefill_tokens
                    )
                )
                running.append(seq)
        else:
            assert self._is_decode()
            # TODO: schedule decodes. What happens if we can't fit the next token in the batch? Sarathi preempts lowest priority request
            # Here I don't think we can do that since preeemption would cause it to get prefilled again
            # Swaps might also get complicated
            # TODO: right now we're using self.swapped_out when pulling deocde requests back in. Let's make it separate?

        # make sure that prefills are at the start of the batch, so that we don't violate assumptions
        # made in the original vllm codebase
        self.running = running

        return SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=ignored_seq_ids,
            preempted_seq_ids=preempted_seq_ids,
            begin_swap_in_seq_ids=[],
            begin_swap_out_seq_ids=[],
            scheduled_seq_metadata_list=scheduled_seq_metadata_list,
        )
