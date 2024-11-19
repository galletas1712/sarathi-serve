from dataclasses import dataclass
from typing import List

from sarathi.core.datatypes.sequence import SequenceScheduleMetadata


@dataclass(frozen=True)
class SchedulerOutputs:

    id: int
    ignored_seq_ids: List[str]
    preempted_seq_ids: List[str]
    swap_out_seq_ids: List[str]
    swap_out_lens: List[int]
    swap_in_seq_ids: List[str]
    scheduled_seq_id_metadata_list: List[SequenceScheduleMetadata]

    def is_empty(self) -> bool:
        # Used to check if we should run execute_model at all (but that includes cache swapping)
        # NOTE: pipeline_parallel_engine has a different definition and this is invalid
        return not self.scheduled_seq_id_metadata_list and not self.swap_in_seq_ids and not self.swap_out_seq_ids