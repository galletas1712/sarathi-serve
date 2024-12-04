from dataclasses import dataclass
from typing import List, Optional

from sarathi.metrics.types.batch_metrics import BatchMetrics


@dataclass
class SequenceMetrics:
    seq_id: str
    arrival_timestamp: float
    batch_ids_scheduled: List[int]
    TBTs: List[float]
    arrival_to_scheduled_delay: Optional[float]

    num_prompt_tokens: int
    num_output_tokens: int

    is_prefill: bool
    last_prefill_batch_id: Optional[int]
    prefill_done_to_first_decode_delay: Optional[float]
    arrival_to_first_decode_delay: Optional[float]

    next_tbt_offset: float
    total_offset: float

    def __init__(self, seq_id: str, arrival_timestamp: float):
        self.seq_id = seq_id
        self.arrival_timestamp = arrival_timestamp
        self.batch_ids_scheduled: List[int] = []
        self.TBTs: List[float] = []

        self.num_prompt_tokens = 0
        self.num_output_tokens = 0

        self.is_prefill = True
        self.last_prefill_batch_id = None
        self.prefill_done_to_first_decode_delay = None
        self.arrival_to_first_decode_delay = None

        self.next_tbt_offset = 0
        self.total_offset = 0

    def schedule(
        self,
        batch_id: int,
        scheduled_timestamp: float,
        num_prompt_tokens: int,
        num_output_tokens: int,
    ):
        if not self.batch_ids_scheduled:
            assert self.total_offset == self.next_tbt_offset
            self.arrival_to_scheduled_delay = (
                scheduled_timestamp - self.arrival_timestamp - self.total_offset
            )
        self.batch_ids_scheduled.append(batch_id)
        self.num_prompt_tokens += num_prompt_tokens
        self.num_output_tokens += num_output_tokens


@dataclass
class ProcessedSeqMetrics:
    num_prompt_tokens: int
    num_output_tokens: int
    num_batches: int
    num_decode_batches: int
    end_to_end_time: float
    decode_time: float
    arrival_to_scheduled_delay: float
    arrival_to_first_decode_delay: float
    tbts_raw: List[float]

    @staticmethod
    def create(
        seq_metrics: SequenceMetrics, batch_metrics: List[BatchMetrics]
    ) -> "ProcessedSeqMetrics":
        # Find first decode batch
        first_decode_batch = None
        for batch_id in seq_metrics.batch_ids_scheduled:
            if not batch_metrics[batch_id].prefill_batched_tokens:
                first_decode_batch = batch_id
                break
        assert first_decode_batch is not None

        processed_seq_metrics = ProcessedSeqMetrics(
            num_prompt_tokens=seq_metrics.num_prompt_tokens,
            num_output_tokens=seq_metrics.num_output_tokens,
            num_batches=len(seq_metrics.batch_ids_scheduled),
            num_decode_batches=sum(
                1
                for batch_id in seq_metrics.batch_ids_scheduled
                if not batch_metrics[batch_id].prefill_batched_tokens
            ),
            arrival_to_scheduled_delay=seq_metrics.arrival_to_scheduled_delay,
            arrival_to_first_decode_delay=seq_metrics.arrival_to_first_decode_delay,
            end_to_end_time=(
                batch_metrics[seq_metrics.batch_ids_scheduled[-1]].end_timestamp
                - seq_metrics.arrival_timestamp
                - seq_metrics.total_offset
            ),
            decode_time=(
                batch_metrics[seq_metrics.batch_ids_scheduled[-1]].end_timestamp
                - batch_metrics[seq_metrics.batch_ids_scheduled[0]].start_timestamp
            ),
            tbts_raw=seq_metrics.TBTs,
        )

        return processed_seq_metrics
