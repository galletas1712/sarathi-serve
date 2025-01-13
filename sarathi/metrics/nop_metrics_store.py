from typing import Any, Dict, List

from sarathi.core.datatypes.sequence import SequenceExecutionMetadata


class WorkerMetricsStore:
    """
    No-op version of the WorkerMetricsStore that maintains the same interface
    but doesn't collect any metrics. Useful for testing or when metrics
    collection needs to be disabled.
    """

    def __init__(self, disagg_emulation: bool):
        pass

    def request_arrived(self, seq_id: str, arrival_timestamp: float):
        pass

    def on_batch_start(self, batch_id: int):
        pass

    def on_batch_scheduled(
        self, batch_id: int, seq_exec_metadata_list: List[SequenceExecutionMetadata]
    ):
        pass

    def on_batch_end(self, batch_id: int, finished_seq_ids: List[str]):
        pass

    def add_engine_scheduler_latency(self, latency: float):
        pass

    def mark_initial_memory_profiling_done(self):
        pass

    def reset(self):
        pass

    def process_metrics(self) -> Dict[str, Any]:
        return {"global_metrics": {}, "seq_metrics": {}}
