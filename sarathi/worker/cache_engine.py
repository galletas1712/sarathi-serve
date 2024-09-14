"""CacheEngine class for managing the KV cache."""

from typing import Dict, List, Tuple, Union

import torch

from sarathi.config import ModelConfig, ParallelConfig, SystemConfig
from sarathi.logger import init_logger
from sarathi.model_executor.attention import get_attention_wrapper

logger = init_logger(__name__)

KVCache = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU KV cache.
    """

    def __init__(
        self,
        config: SystemConfig,
    ) -> None:
        self.head_size = config.model_config.get_head_size()
        self.num_layers = config.model_config.get_num_layers(config.parallel_config)
        self.num_heads = config.model_config.get_num_kv_heads(config.parallel_config)
        self.dtype = config.model_config.dtype

        self.block_size = config.cache_config.block_size
        self.num_gpu_blocks = config.cache_config.num_gpu_blocks
        self.num_cpu_blocks = config.cache_config.num_cpu_blocks

        assert self.num_gpu_blocks is not None
        assert self.num_cpu_blocks is not None

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(self.num_gpu_blocks, "cuda")
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

        self.finish_swap_out_events = {}
        self.finish_swap_in_events = {}

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache: List[torch.Tensor] = []
        num_blocks_per_layer = (num_blocks + self.num_layers - 1) // self.num_layers
        for _ in range(self.num_layers):
            kv_cache.append(
                get_attention_wrapper().get_cache_block(
                    num_blocks_per_layer,
                    dtype=self.dtype,
                    pin_memory=(device == "cpu"),
                    device=device)
                )
        return kv_cache
    
    def _begin_swap(self, swap_mapping: Dict[str, List[Tuple[int, int]]], swap_in: bool) -> None:
        for seq_id, src_to_dst in swap_mapping.items():
            src_to_dst = torch.tensor(src_to_dst, dtype=torch.int64, device="cpu")
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                finish_event = torch.cuda.Event()
                for i in range(self.num_layers):
                    if swap_in:
                        get_attention_wrapper().swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                                    src_to_dst)
                    else:
                        get_attention_wrapper().swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                                    src_to_dst)
                finish_event.record()
            if swap_in:
                self.finish_swap_in_events[seq_id] = finish_event
            else:
                self.finish_swap_out_events[seq_id] = finish_event

    def begin_swap_in(self, swap_mapping: Dict[str, List[Tuple[int, int]]]) -> None:
        self._begin_swap(swap_mapping, swap_in=True)

    def begin_swap_out(self, src_to_dst: List[Tuple[int, int]]) -> torch.cuda.Event:
        self._begin_swap(src_to_dst, swap_in=False)

    def pop_finished(self) -> Tuple[List[str], List[str]]:
        finished_swap_in_seq_ids = []
        finished_swap_out_seq_ids = []
        logger.debug(f"Swap in events: {list(self.finish_swap_in_events.items())}")
        logger.debug(f"Swap out events: {list(self.finish_swap_out_events.items())}")

        for seq_id, event in self.finish_swap_in_events.items():
            if event.query():
                finished_swap_in_seq_ids.append(seq_id)
            else:
                logger.debug(f"Event for swap in {seq_id} not done")
        
        for seq_id in finished_swap_in_seq_ids:
            del self.finish_swap_in_events[seq_id]

        for seq_id, event in self.finish_swap_out_events.items():
            if event.query():
                finished_swap_out_seq_ids.append(seq_id)
            else:
                logger.debug(f"Event for swap out {seq_id} not done")
        
        for seq_id in finished_swap_out_seq_ids:
            del self.finish_swap_out_events[seq_id]
        
        return finished_swap_in_seq_ids, finished_swap_out_seq_ids

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        dtype_size = _get_dtype_size(model_config.dtype)
        return dtype_size * total


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
