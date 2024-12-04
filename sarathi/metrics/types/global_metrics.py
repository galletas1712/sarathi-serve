from dataclasses import dataclass
from typing import Dict, List

from sarathi.metrics.types.batch_metrics import BatchMetrics
from sarathi.metrics.types.seq_metrics import ProcessedSeqMetrics


def extract_key_from_all_processed_seq_metrics(
    all_processed_seq_metrics: Dict[str, ProcessedSeqMetrics], key_getter
):
    return [
        key_getter(processed_seq_metrics)
        for processed_seq_metrics in all_processed_seq_metrics.values()
    ]


def flatten_key_from_all_processed_seq_metrics(
    all_processed_seq_metrics: Dict[str, ProcessedSeqMetrics], key_getter
):
    ret = []
    for processed_seq_metrics in all_processed_seq_metrics.values():
        ret.extend(key_getter(processed_seq_metrics))
    return ret


@dataclass
class GlobalMetrics:
    qps: float
    tbts_raw: List[float]
    prefill_done_to_first_decode_delays_raw: List[float]
    arrival_to_first_scheduled_delays_raw: List[float]
    arrival_to_first_decode_delays_raw: List[float]
    end_to_end_times_raw: List[float]
    engine_scheduler_latencies_raw: List[float]

    @staticmethod
    def create(
        all_processed_seq_metrics: Dict[str, ProcessedSeqMetrics],
        batch_metrics: BatchMetrics,
        engine_scheduler_latencies: List[float],
    ) -> "GlobalMetrics":
        global_metrics = GlobalMetrics(
            qps=len(all_processed_seq_metrics)
            / (batch_metrics[-1].end_timestamp - batch_metrics[0].start_timestamp),
            tbts_raw=flatten_key_from_all_processed_seq_metrics(
                all_processed_seq_metrics, lambda x: x.tbts_raw
            ),
            prefill_done_to_first_decode_delays_raw=extract_key_from_all_processed_seq_metrics(
                all_processed_seq_metrics, lambda x: x.arrival_to_first_decode_delay
            ),
            arrival_to_first_scheduled_delays_raw=extract_key_from_all_processed_seq_metrics(
                all_processed_seq_metrics, lambda x: x.arrival_to_scheduled_delay
            ),
            arrival_to_first_decode_delays_raw=extract_key_from_all_processed_seq_metrics(
                all_processed_seq_metrics, lambda x: x.arrival_to_first_decode_delay
            ),
            end_to_end_times_raw=extract_key_from_all_processed_seq_metrics(
                all_processed_seq_metrics, lambda x: x.end_to_end_time
            ),
            engine_scheduler_latencies_raw=engine_scheduler_latencies,
        )
        return global_metrics
