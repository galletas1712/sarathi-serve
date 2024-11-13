from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from sarathi.config import SystemConfig
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import (
    SamplerOutput,
    SamplerOutputs,
    Sequence,
    SequenceScheduleMetadata,
)
from sarathi.core.datatypes.sequence_status import SequenceStatus
from sarathi.utils.threading_utils import synchronized


class BaseSequenceManager(ABC):

    def __init__(self, config: SystemConfig):
        self.config = config
        self.seq_map: Dict[str, Sequence] = {}

    @synchronized
    def add_seq(self, seq: Sequence) -> None:
        assert seq.seq_id not in self.seq_map
        self.seq_map[seq.seq_id] = seq

    def _free_seq(self, seq_id: str) -> None:
        del self.seq_map[seq_id]
    
    def _ignore_seq(self, seq_id: str) -> None:
        self.seq_map[seq_id].set_status(SequenceStatus.FINISHED_IGNORED)
        self._free_seq(seq_id)

    def _preempt_seq(self, seq_id: str) -> None:
        # Sets to WAITING inside
        self.seq_map[seq_id].reset_for_recompute()

    def _swap_out_seq(self, seq_id: str, num_blocks_to_swap: Optional[int] = None) -> None:
        self.seq_map[seq_id].set_status(SequenceStatus.SWAPPED_OUT)
    
    def _begin_swap_in_seq(self, seq_id: str) -> None:
        self.seq_map[seq_id].set_status(SequenceStatus.SWAPPING_IN)
    
    def _finish_swap_in_seq(self, seq_id: str) -> None:
        self.seq_map[seq_id].set_status(SequenceStatus.PAUSED)

    def _pause_seq(self, seq_id: str) -> None:
        self.seq_map[seq_id].set_status(SequenceStatus.PAUSED)

    def _resume_seq(self, seq_id: str) -> None:
        self.seq_map[seq_id].set_status(SequenceStatus.RUNNING)

    def _on_seq_scheduled(self, seq_id_metadata: SequenceScheduleMetadata) -> None:
        assert seq_id_metadata.seq_id in self.seq_map
        self._resume_seq(seq_id_metadata.seq_id)

    @synchronized
    def on_schedule(
        self,
        scheduler_outputs: SchedulerOutputs,
    ) -> None:
        assert (
            len(scheduler_outputs.swap_out_lens) == len(scheduler_outputs.swap_out_seq_ids) or
            len(scheduler_outputs.swap_out_lens) == 0
        )  # NOTE: Either we specify the number of blocks to swap out or we don't
        for seq_id in scheduler_outputs.ignored_seq_ids:
            self._ignore_seq(seq_id)

        for seq_id in scheduler_outputs.preempted_seq_ids:
            self._preempt_seq(seq_id)
        
        # Swap out
        if scheduler_outputs.swap_out_lens:
            for seq_id, num_blocks_to_swap in zip(
                scheduler_outputs.swap_out_seq_ids,
                scheduler_outputs.swap_out_lens,
            ):
                self._swap_out_seq(seq_id, num_blocks_to_swap)
        else:
            for seq_id in scheduler_outputs.swap_out_seq_ids:
                self._swap_out_seq(seq_id)
        
        for seq_id in scheduler_outputs.begin_swap_in_seq_ids:
            self._begin_swap_in_seq(seq_id)
        
        for seq_id_metadata in scheduler_outputs.scheduled_seq_id_metadata_list:
            self._on_seq_scheduled(seq_id_metadata)  # This will call _resume_seq inside

    @abstractmethod
    def _on_append_token(self, seq: Sequence) -> None:
        pass

    def _process_seq_output(
        self,
        seq: Sequence,
        sample: SamplerOutput,
    ) -> bool:
        # at this point, the seq should be in paused state
        assert not seq.is_finished()

        if not seq.is_prompt_processing_finished():
            return

        seq.append_token_id(sample.output_token)
        self._on_append_token(seq)

        # this function will update the seq status
        # to finished if the stop condition is met
        seq.check_stop()
        if seq.is_finished():
            self._free_seq(seq.seq_id)
            return True
        
        return False

    @synchronized
    def on_step_completed(
        self,
        scheduler_outputs: SchedulerOutputs,
        sampler_outputs: Optional[SamplerOutputs],
    ) -> List[str]:
        finished_seq_ids = []

        for seq_id_metadata, sampler_output in zip(
            scheduler_outputs.scheduled_seq_id_metadata_list, sampler_outputs
        ):
            seq_id = seq_id_metadata.seq_id
            assert seq_id == sampler_output.seq_id
            seq = self.seq_map[seq_id]
            if seq.is_waiting() or seq.is_swapped_out():
                # seq is preempted
                # this can happen with pipeline parallel -- if the system
                # runs out of memory, it will preempt the last arrived request
                # this request might still be executing when the next stage scheduling
                # triggers the preemption
                continue

            if not seq.is_prompt_processing_finished():
                seq.update_prompt_tokens_processed(
                    seq_id_metadata.prompt_chunk_len
                )

            self._pause_seq(seq_id)

            finished = self._process_seq_output(
                seq,
                sampler_output,
            )
            if finished:
                finished_seq_ids.append(seq_id)
        
        return finished_seq_ids

    @synchronized
    def mark_swap_in_finished(self, finished_swap_in_seq_ids: List[str]) -> None:
        for seq_id in finished_swap_in_seq_ids:
            self._finish_swap_in_seq(seq_id)