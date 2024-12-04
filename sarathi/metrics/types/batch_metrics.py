from dataclasses import dataclass
from typing import Optional


@dataclass
class BatchMetrics:
    """Metrics for a singular batch of requests."""

    batch_id: int
    start_timestamp: float
    scheduled_timestamp: Optional[float]
    end_timestamp: Optional[float]
    prefill_kv_cache_tokens: Optional[int]
    prefill_batched_tokens: Optional[int]
    decode_kv_cache_tokens: Optional[int]
    decode_batched_tokens: Optional[int]
    num_requests: Optional[int]

    def __init__(self, batch_id: int, start_timestamp: float):
        self.batch_id = batch_id
        self.start_timestamp = start_timestamp

    def schedule(
        self,
        scheduled_timestamp: float,
        prefill_kv_cache_tokens: int,
        prefill_batched_tokens: int,
        decode_kv_cache_tokens: int,
        decode_batched_tokens: int,
        num_requests: int,
    ):
        self.scheduled_timestamp = scheduled_timestamp
        self.prefill_kv_cache_tokens = prefill_kv_cache_tokens
        self.prefill_batched_tokens = prefill_batched_tokens
        self.decode_kv_cache_tokens = decode_kv_cache_tokens
        self.decode_batched_tokens = decode_batched_tokens
        self.num_requests = num_requests

    def end(self, end_timestamp: float):
        self.end_timestamp = end_timestamp
