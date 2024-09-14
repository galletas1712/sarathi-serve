from typing import List

from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata


class SchedulerOutputs:

    def __init__(
        self,
        id: int,
        ignored_seq_ids: List[str],
        preempted_seq_ids: List[str],
        begin_swap_out_seq_ids: List[str],
        begin_swap_in_seq_ids: List[str],
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata],
    ) -> None:
        self.id = id
        self.ignored_seq_ids = ignored_seq_ids
        self.preempted_seq_ids = preempted_seq_ids
        self.begin_swap_out_seq_ids = begin_swap_out_seq_ids
        self.begin_swap_in_seq_ids = begin_swap_in_seq_ids
        self.scheduled_seq_metadata_list = sorted(
            scheduled_seq_metadata_list, key=lambda x: not x.is_prompt  # NOTE: This is sorting decodes at the beginning
        )
        self.prompt_chunk_lens = [
            metadata.num_prompt_tokens for metadata in scheduled_seq_metadata_list
        ]
        self.num_batched_prompt_tokens = sum(self.prompt_chunk_lens)
        self.num_batched_output_tokens = sum(
            metadata.num_output_tokens for metadata in scheduled_seq_metadata_list
        )
        self.num_batched_tokens = sum(
            metadata.num_tokens for metadata in scheduled_seq_metadata_list
        )

    def is_empty(self) -> bool:
        # Used to check if we should run execute_model at all (but that includes cache swapping)
        # NOTE: pipeline_parallel_engine has a different definition and this is invalid
        return not self.scheduled_seq_metadata_list and not self.begin_swap_in_seq_ids and not self.begin_swap_out_seq_ids

    def has_no_output(self) -> bool:
        # NOTE: same deal with pipeline_parallel_engine
        return not self.scheduled_seq_metadata_list

    @property
    def seq_ids(self) -> List[str]:
        return [metadata.seq_id for metadata in self.scheduled_seq_metadata_list]

    def __repr__(self) -> str:
        return (
            f"SchedulerOutputs(id={self.id}, "
            f"ignored_seq_ids={self.ignored_seq_ids}, "
            f"preempted_seq_ids={self.preempted_seq_ids}, "
            f"begin_swap_out_seq_ids={self.begin_swap_out_seq_ids}, "
            f"begin_swap_in_seq_ids={self.begin_swap_in_seq_ids}, "
            f"scheduled_seq_metadata_list={self.scheduled_seq_metadata_list})"
        )
