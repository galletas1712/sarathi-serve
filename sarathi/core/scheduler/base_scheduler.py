from abc import ABC, abstractmethod
from typing import Dict, List

from sarathi.config import BaseSchedulerConfig, CacheConfig, ModelConfig, ParallelConfig
from sarathi.core.block_space_manager.block_space_manager_registry import (
    BlockSpaceManagerRegistry,
)
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceStatus
from sarathi.core.policy import PolicyFactory
from sarathi.logger import init_logger
from sarathi.metrics.metrics_store import MetricsStore

logger = init_logger(__name__)


class BaseScheduler(ABC):

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: BaseSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        self.metrics_store = MetricsStore.get_instance()
        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config

        # we maintain this just for logging purposes
        self._iteration_id = -1

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Create the block space manager.
        self.block_manager = BlockSpaceManagerRegistry.get(
            scheduler_config.get_type(),
            cache_config.block_size,
            cache_config.num_gpu_blocks,
            cache_config.num_cpu_blocks,
            model_config.max_model_len,
        )
        self.prompt_limit = model_config.max_model_len

        # number of running batches should be less than or equal to the number of pipeline stages
        self.num_running_batches = 0

        # TODO(zhuohan): Use deque instead of list for better performance.
        # Sequence groups in the WAITING state.
        self.waiting: List[Sequence] = []
        # Sequence groups in the RUNNING state.
        self.running: List[Sequence] = []

        self.swapping_out: Dict[str, Sequence] = {}
        self.swapped_out: Dict[str, Sequence] = {}
        self.swapping_in: Dict[str, Sequence] = {}
        self.swapped_in: Dict[str, Sequence] = {}

    def reset_state(self) -> None:
        self._iteration_id = -1

    def add_seq(self, seq: Sequence) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq)

    def has_unfinished_seqs(self) -> bool:
        logger.debug(f"Iteration: {self._iteration_id}, waiting: {self.waiting}, running: {self.running}, swapping_out: {self.swapping_out}, swapped_out: {self.swapped_out}, swapping_in: {self.swapping_in}, swapped_in: {self.swapped_in}")
        return self.waiting or self.running or self.swapping_out or self.swapped_out or self.swapping_in or self.swapped_in

    def get_num_unfinished_seqs(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapping_out) + len(self.swapped_out) + len(self.swapping_in) + len(self.swapped_in)

    @abstractmethod
    def _schedule(self) -> SchedulerOutputs:
        pass

    def schedule(self) -> SchedulerOutputs:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running and self.waiting.
        self._iteration_id += 1

        if self.num_running_batches >= self.parallel_config.pipeline_parallel_size:
            return SchedulerOutputs(
                self._iteration_id,
                ignored_seq_ids=[],
                preempted_seq_ids=[],
                begin_swap_in_seq_ids=[],
                begin_swap_out_seq_ids=[],
                scheduled_seq_metadata_list=[],
            )

        scheduler_outputs = self._schedule()

        if not scheduler_outputs.is_empty():
            self.num_running_batches += 1
        
        return scheduler_outputs

    def free_finished_seqs(self) -> None:
        for seq in self.running:
            if seq.is_finished():
                self._free_seq(seq)
        self.running = [seq for seq in self.running if not seq.is_finished()]

    def on_step_completed(self) -> None:
        self.free_finished_seqs()
        self.num_running_batches -= 1

    def _allocate(self, seq: Sequence) -> None:
        self.block_manager.allocate(seq)

    def _free_seq(self, seq: Sequence) -> None:
        self.block_manager.free(seq.seq_id)

    def _append_slot(
        self,
        seq: Sequence,
    ) -> None:
        # TODO: this was an assertion, but there's actually a one iteration delay just because of the way we refactored things
        # assert seq.is_executing()
        self.block_manager.append_slot(seq)

    def _preempt(
        self,
        seq: Sequence,
    ) -> None:
        assert seq.is_executing()
        self._free_seq(seq)
        self.waiting.insert(0, seq)
    
    def _begin_swap_in(self, seq: Sequence) -> None:
        assert seq.is_swapped_out()
        del self.swapped_out[seq.seq_id]
        self.swapping_in[seq.seq_id] = seq
        self.block_manager.begin_swap_in(seq.seq_id)
    
    def _finish_swap_in(self, seq: Sequence) -> None:
        assert seq.is_swapping_in()
        del self.swapping_in[seq.seq_id]
        self.swapped_in[seq.seq_id] = seq
        self.block_manager.finish_swap_in(seq.seq_id)
    
    def _begin_swap_out(self, seq: Sequence) -> None:
        assert seq.is_executing()
        if seq.seq_id in self.swapped_in:
            logger.warning(f"Sequence {seq.seq_id} to swap in was recently swapped out and not yet made progress")
            del self.swapped_in[seq.seq_id]  # NOTE: Maybe we didn't remove from swapped_in queue properly
        self.swapping_out[seq.seq_id] = seq
        self.block_manager.begin_swap_out(seq.seq_id)
    
    def _finish_swap_out(self, seq: Sequence) -> None:
        assert seq.is_swapping_out()
        del self.swapping_out[seq.seq_id]
        self.swapped_out[seq.seq_id] = seq
        self.block_manager.finish_swap_out(seq.seq_id)

    def _check_request_prompt_length(self, seq: Sequence) -> bool:
        if seq.get_len() > self.prompt_limit:
            logger.warning(
                f"Input prompt ({seq.get_len()} tokens) is too long"
                f" and exceeds limit of {self.prompt_limit}"
            )
            seq.set_status(SequenceStatus.FINISHED_IGNORED)
            self.waiting.pop(0)
            return False

        return True

    def mark_finished(self, finished_swap_in_seq_ids: List[str], finished_swap_out_seq_ids: List[str]) -> None:
        for seq_id in finished_swap_in_seq_ids:
            logger.debug(f"Sequence {seq_id} has finished swapping in")
            seq = self.swapping_in[seq_id]
            self._finish_swap_in(seq)
        
        for seq_id in finished_swap_out_seq_ids:
            logger.debug(f"Sequence {seq_id} has finished swapping out")
            seq = self.swapping_out[seq_id]
            self._finish_swap_out(seq)