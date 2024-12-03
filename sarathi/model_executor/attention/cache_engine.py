"""CacheEngine class for managing the KV cache."""

from typing import Dict, List, Optional, Tuple, Union

import torch

from sarathi.config import CacheConfig, ModelConfig, ParallelConfig
from sarathi.logger import init_logger
from sarathi_kernels.cache_ops import swap_blocks

logger = init_logger(__name__)

KVCache = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU KV cache.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        device: torch.device,
    ) -> None:
        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)
        self.dtype = model_config.dtype

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        self.duplicate_kv_cache = cache_config.duplicate_kv_cache
        self.async_swap_in = cache_config.async_swap_in

        assert self.num_gpu_blocks is not None
        assert self.num_cpu_blocks is not None

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(self.num_gpu_blocks, device)
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")
        self.swap_in_stream = torch.cuda.Stream(device=device)
        self.duplicate_stream = torch.cuda.Stream(device=device)

        self.finish_swap_in_events: dict[int, torch.cuda.Event] = {}

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache: List[torch.Tensor] = []
        for _ in range(self.num_layers):
            kv_cache.append(
                torch.empty(
                    num_blocks,
                    2,
                    self.block_size,
                    self.num_heads,
                    self.head_size,
                    dtype=self.dtype,
                    pin_memory=(device == "cpu"),
                    device=device,
                )
            )
        return kv_cache

    def begin_swap_in(self, swap_mapping: dict[str, list[tuple[int, int]]]) -> None:
        # Create events for each layer and sync them one by one, but batching the sync of all seqs at a time for each layer
        with torch.cuda.stream(self.swap_in_stream):
            for layer_id in range(self.num_layers):
                for _, src_to_dst in swap_mapping.items():
                    src_to_dst = torch.tensor(
                        src_to_dst, dtype=torch.int64, device="cpu"
                    )
                    swap_blocks(
                        self.cpu_cache[layer_id], self.gpu_cache[layer_id], src_to_dst
                    )
                layer_finish_event = torch.cuda.Event()
                layer_finish_event.record()
                self.finish_swap_in_events[layer_id] = layer_finish_event

    def wait_for_swap_in(self, layer: Optional[int] = None) -> None:
        assert layer is None or (layer >= 0 and layer < self.num_layers)
        if layer is None:
            layer = self.num_layers - 1
        self.finish_swap_in_events[layer].synchronize()
        assert self.finish_swap_in_events[layer].query()
        del self.finish_swap_in_events[layer]

        # TODO: remove later
        for prev_layer in range(layer):
            assert self.finish_swap_in_events[prev_layer].query()

        if layer == self.num_layers - 1:
            self.finish_swap_in_events.clear()

    def swap_in(self, swap_mapping: dict[str, list[tuple[int, int]]]) -> None:
        self.begin_swap_in(swap_mapping=swap_mapping)
        self.wait_for_swap_in()
        assert not self.finish_swap_in_events

    ########## Vanilla swap out

    def swap_out(self, swap_mapping: Dict[str, List[Tuple[int, int]]]) -> None:
        assert not self.duplicate_kv_cache
        finish_event = torch.cuda.Event()
        for _, src_to_dst in swap_mapping.items():
            src_to_dst = torch.tensor(src_to_dst, dtype=torch.int64, device="cpu")
            for i in range(self.num_layers):
                swap_blocks(self.gpu_cache[i], self.cpu_cache[i], src_to_dst)
        finish_event.record()  # NOTE: This should be the default stream
        finish_event.synchronize()

    ########## Methods for handling duplicate KV cache swap out

    def register_duplicate_mapping_optional(
        self, duplicate_mapping: Optional[list[tuple[int, int]]]
    ) -> None:
        if self.duplicate_kv_cache:
            assert duplicate_mapping is not None
            assert not hasattr(self, "duplicate_mapping")
            self.duplicate_mapping = torch.tensor(
                duplicate_mapping, dtype=torch.long, device="cpu"
            )
        else:
            assert duplicate_mapping is None

    def sync_and_remove_duplicate_mapping_optional(self) -> None:
        if self.duplicate_kv_cache:
            assert hasattr(self, "duplicate_mapping")
            self.duplicate_stream.synchronize()
            del self.duplicate_mapping

    def swap_out_duplicate_kv_cache(self, layer_id: int) -> None:
        if self.duplicate_kv_cache:
            assert layer_id is not None
            assert (
                hasattr(self, "duplicate_mapping")
                and self.duplicate_mapping is not None
            )

            with torch.cuda.stream(self.duplicate_stream):
                swap_blocks(
                    self.gpu_cache[layer_id],
                    self.cpu_cache[layer_id],
                    self.duplicate_mapping,
                )

    ######### Misc methods

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
