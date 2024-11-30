"""A block manager that manages token blocks, but without materializing any block tables - just a dry run for the scheduler to simulate what happens."""

import enum
from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import chain
from typing import Dict, List, Optional, Tuple, Union, Iterator

from sarathi.core.datatypes.sequence import Sequence


class BlockDevice(enum.Enum):
    GPU = "gpu"
    CPU = "cpu"


class BaseBlockAllocator:
    """Manages the number of free physical token blocks for a device."""

    def __init__(
        self,
        num_blocks: int,
        watermark: Optional[float] = None,
    ) -> None:
        self.__num_blocks = num_blocks - (
            0 if watermark is None else int(watermark * num_blocks)
        )
        self.__num_blocks_allocated = 0

    @property
    def num_total_blocks(self) -> int:
        return self.__num_blocks

    @property
    def num_blocks_allocated(self) -> int:
        return self.__num_blocks_allocated

    @property
    def num_free_blocks(self) -> int:
        return self.__num_blocks - self.__num_blocks_allocated

    def can_allocate(self, num_blocks: int) -> bool:
        return self.num_free_blocks >= num_blocks

    def allocate(self, num_blocks: int) -> int:
        if not self.can_allocate(num_blocks):
            raise ValueError("Out of memory! Not enough free blocks are available.")
        self.__num_blocks_allocated += num_blocks
        return num_blocks

    def free(self, blocks: int) -> None:
        if blocks > self.__num_blocks_allocated:
            raise ValueError("Invalid number of blocks to free.")
        self.__num_blocks_allocated -= blocks


class DryRunBlockAllocator(BaseBlockAllocator):
    pass


class BlockAllocator(BaseBlockAllocator):
    def __init__(self, num_blocks: int, watermark: Optional[float] = None) -> None:
        super().__init__(num_blocks, watermark=watermark)
        self.__free_blocks = list(range(self.num_total_blocks))

    def allocate(self, num_blocks: int) -> List[int]:
        super().allocate(num_blocks)
        result = deepcopy(self.__free_blocks[-num_blocks:])
        del self.__free_blocks[-num_blocks:]
        self.__ensure_consistent_with_parent()
        return result

    def free(self, blocks: List[int]) -> None:
        super().free(len(blocks))
        self.__free_blocks.extend(blocks)
        self.__ensure_consistent_with_parent()

    def __ensure_consistent_with_parent(self):
        assert self.num_free_blocks == len(self.__free_blocks)


class BaseBlockSpaceManager(ABC):
    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        duplicate_kv_cache: bool,
        max_model_len: int,
        watermark: float = 0.01,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.duplicate_kv_cache = duplicate_kv_cache
        self.max_model_len = max_model_len

        assert watermark >= 0.0

        self._allocators = {
            BlockDevice.GPU: self._init_allocator(num_gpu_blocks, watermark=watermark),
            BlockDevice.CPU: self._init_allocator(num_cpu_blocks, watermark=None),
        }

        # Mapping: seq_id -> device -> available blocks
        self._block_tables: Dict[str, Dict[BlockDevice, Union[List[int], int]]] = {}
        self._curr_seq_blocks_swapped_out: Dict[str, int] = {}

    @abstractmethod
    def _init_allocator(
        self, num_blocks: int, watermark: Optional[float] = None
    ) -> Union[DryRunBlockAllocator, BlockAllocator]:
        raise NotImplementedError

    ########## Read-only methods ##########

    def get_num_free_blocks(self, device: BlockDevice) -> int:
        return self._allocators[device].num_free_blocks

    def get_seq_num_blocks_allocated(self, seq_id: str, device: BlockDevice) -> int:
        assert isinstance(seq_id, str)
        if device not in self._block_tables[seq_id]:
            return 0
        assert isinstance(self._block_tables[seq_id][device], list) or isinstance(
            self._block_tables[seq_id][device], int
        )
        return (
            len(self._block_tables[seq_id][device])
            if isinstance(self._block_tables[seq_id][device], list)
            else self._block_tables[seq_id][device]
        )

    def is_allocated_in_gpu(self, seq_id: str) -> bool:
        assert isinstance(seq_id, str)
        return BlockDevice.GPU in self._block_tables[seq_id]

    def is_allocated_in_cpu(self, seq_id: str) -> bool:
        assert isinstance(seq_id, str)
        return BlockDevice.CPU in self._block_tables[seq_id]

    def _can_allocate(self, seq: Sequence, device: BlockDevice) -> bool:
        """Check if we can allocate memory for the entire sequence."""
        assert isinstance(seq, Sequence)
        return self._allocators[device].can_allocate(seq.get_num_logical_blocks())

    def can_allocate(self, seq: Sequence) -> bool:
        assert isinstance(seq, Sequence)
        assert seq.seq_id not in self._block_tables
        return self._can_allocate(seq, BlockDevice.GPU) and (
            self.duplicate_kv_cache or self._can_allocate(seq, BlockDevice.CPU)
        )

    def _can_append_slot(self, seq: Sequence, device: BlockDevice) -> bool:
        assert isinstance(seq, Sequence)
        assert (
            seq.seq_id in self._block_tables
            and device in self._block_tables[seq.seq_id]
        )
        num_blocks_to_allocate = (
            seq.get_num_logical_blocks()
            - self.get_seq_num_blocks_allocated(seq.seq_id, device)
        )
        assert num_blocks_to_allocate <= 1
        return self._allocators[device].can_allocate(num_blocks_to_allocate)

    def can_append_slot(self, seq: Sequence) -> bool:
        assert isinstance(seq, Sequence)
        assert not self.duplicate_kv_cache or self.get_seq_num_blocks_allocated(
            seq.seq_id, BlockDevice.GPU
        ) == self.get_seq_num_blocks_allocated(seq.seq_id, BlockDevice.CPU)
        result = self._can_append_slot(seq, BlockDevice.GPU)
        if result and self.duplicate_kv_cache:
            assert self._can_append_slot(seq, BlockDevice.CPU)
        return result

    def can_swap_in(self, seq_id: str) -> bool:
        assert isinstance(seq_id, str)
        assert BlockDevice.CPU in self._block_tables[seq_id]
        assert seq_id in self._curr_seq_blocks_swapped_out
        if not self.duplicate_kv_cache:
            assert (
                self.get_seq_num_blocks_allocated(seq_id, BlockDevice.CPU)
                == self._curr_seq_blocks_swapped_out[seq_id]
            )
        return (
            self.get_num_free_blocks(BlockDevice.GPU)
            >= self._curr_seq_blocks_swapped_out[seq_id]
        )

    def can_swap_in_and_append_slot(self, seq_id: str, num_logical_blocks: int) -> bool:
        assert isinstance(seq_id, str)
        assert BlockDevice.CPU in self._block_tables[seq_id]
        assert (
            abs(
                self._curr_seq_blocks_swapped_out[seq_id]
                - (
                    num_logical_blocks
                    - self.get_seq_num_blocks_allocated(seq_id, BlockDevice.GPU)
                )
            )
            <= 1
        )
        result = self.get_num_free_blocks(BlockDevice.GPU) >= (
            num_logical_blocks
            - self.get_seq_num_blocks_allocated(seq_id, BlockDevice.GPU)
        )
        if result:
            assert self.can_swap_in(seq_id)
        return result

    def can_swap_out(self, seq_id: str) -> bool:
        assert isinstance(seq_id, str)
        if self.duplicate_kv_cache:
            return True  # We don't need to allocate anything if it's already in host memory
        assert BlockDevice.GPU in self._block_tables[seq_id]
        return self.get_num_free_blocks(
            BlockDevice.CPU
        ) >= self.get_seq_num_blocks_allocated(seq_id, BlockDevice.GPU)

    ########## Allocations/Swaps ##########

    def allocate(self, seq: Sequence) -> None:
        """Allocates new physical token blocks that will store the prompt tokens. Almost always on GPU."""
        assert isinstance(seq, Sequence)
        assert self.can_allocate(seq)
        self._block_tables[seq.seq_id] = {
            BlockDevice.GPU: self._allocators[BlockDevice.GPU].allocate(
                seq.get_num_logical_blocks()
            )
        }
        if self.duplicate_kv_cache:
            self._block_tables[seq.seq_id][BlockDevice.CPU] = self._allocators[
                BlockDevice.CPU
            ].allocate(seq.get_num_logical_blocks())

    def _append_slot(self, seq: Sequence, device: BlockDevice) -> None:
        """Allocate a physical slot for a new token."""
        assert isinstance(seq, Sequence)
        num_blocks_to_allocate = (
            seq.get_num_logical_blocks()
            - self.get_seq_num_blocks_allocated(seq.seq_id, device)
        )
        assert (
            num_blocks_to_allocate <= 1
        ), f"Can only append one slot at a time. Requested: {num_blocks_to_allocate}. Sequence status: {seq.get_status()}"
        if num_blocks_to_allocate > 0:
            self._block_tables[seq.seq_id][device] += self._allocators[device].allocate(
                num_blocks_to_allocate
            )

    def append_slot(self, seq: Sequence) -> None:
        self._append_slot(seq, BlockDevice.GPU)
        if self.duplicate_kv_cache:
            self._append_slot(seq, BlockDevice.CPU)

    def begin_swap_in(self, seq_id: str):
        assert isinstance(seq_id, str)
        assert self.can_swap_in(seq_id)

        newly_allocated_gpu_blocks = self._allocators[BlockDevice.GPU].allocate(
            self._curr_seq_blocks_swapped_out[seq_id]
        )
        self._block_tables[seq_id][BlockDevice.GPU] = (
            # NOTE: Ordering matters for prefix/suffix - here we reverse the order
            newly_allocated_gpu_blocks + self._block_tables[seq_id][BlockDevice.GPU]
            if BlockDevice.GPU in self._block_tables[seq_id]
            else newly_allocated_gpu_blocks
        )
        self._update_swap_in_mapping(seq_id, self._curr_seq_blocks_swapped_out[seq_id])

    def finish_swap_in(self, seq_id: str):
        assert isinstance(seq_id, str)
        if not self.duplicate_kv_cache:
            self._free_device_blocks(seq_id, BlockDevice.CPU)
        del self._curr_seq_blocks_swapped_out[seq_id]

    def swap_in(self, seq_id: str):
        assert isinstance(seq_id, str)
        assert self.can_swap_in(seq_id)
        self.begin_swap_in(seq_id)
        self.finish_swap_in(seq_id)

    def swap_out(self, seq_id: str, num_blocks_to_swap: Optional[int] = None):
        assert isinstance(seq_id, str)
        assert self.can_swap_out(seq_id)

        num_allocated_blocks = self.get_seq_num_blocks_allocated(
            seq_id, BlockDevice.GPU
        )
        if num_blocks_to_swap is None:
            num_blocks_to_swap = num_allocated_blocks
        assert num_blocks_to_swap <= num_allocated_blocks

        self._curr_seq_blocks_swapped_out[seq_id] = (
            self._curr_seq_blocks_swapped_out.get(seq_id, 0) + num_blocks_to_swap
        )

        if not self.duplicate_kv_cache:
            newly_allocated_cpu_blocks = self._allocators[BlockDevice.CPU].allocate(
                num_blocks_to_swap
            )
            self._block_tables[seq_id][BlockDevice.CPU] = (
                # NOTE: Ordering matters for prefix/suffix
                self._block_tables[seq_id][BlockDevice.CPU] + newly_allocated_cpu_blocks
                if BlockDevice.CPU in self._block_tables[seq_id]
                else newly_allocated_cpu_blocks
            )
            self._update_swap_out_mapping(seq_id, num_blocks_to_swap)
            assert (
                self.get_seq_num_blocks_allocated(seq_id, BlockDevice.CPU)
                == self._curr_seq_blocks_swapped_out[seq_id]
            )

        self._free_device_blocks(
            seq_id, BlockDevice.GPU, num_blocks_to_free=num_blocks_to_swap
        )

    def _update_swap_in_mapping(self, seq_id: str, num_blocks_to_swap: int) -> None:
        raise NotImplementedError

    def _update_swap_out_mapping(self, seq_id: str, num_blocks_to_swap: int) -> None:
        raise NotImplementedError

    ########## Frees ##########

    @abstractmethod
    def _free_device_blocks(
        self, seq_id: str, device: BlockDevice, num_blocks_to_free: Optional[int] = None
    ) -> None:
        raise NotImplementedError

    def _free_block_table(self, seq_id: str) -> None:
        assert isinstance(seq_id, str)
        if seq_id not in self._block_tables:
            # Already freed or haven't been scheduled yet.
            return
        devices = list(self._block_tables[seq_id].keys())
        for device in devices:
            self._free_device_blocks(seq_id, device)
        del self._block_tables[seq_id]

    def free(self, seq_id: str) -> None:
        assert isinstance(seq_id, str)
        self._free_block_table(seq_id)

    def reset(self) -> None:
        for seq_id in self._block_tables.keys():
            self._free_block_table(seq_id)

    ########## Metadata ##########

    def get_block_table_metadata_str(self) -> str:
        gpu_block_table_lens = []
        cpu_block_table_lens = []
        for seq_id in self._block_tables.keys():
            if BlockDevice.GPU in self._block_tables[seq_id]:
                gpu_block_table_lens.append(
                    (seq_id, self.get_seq_num_blocks_allocated(seq_id, BlockDevice.GPU))
                )
            if BlockDevice.CPU in self._block_tables[seq_id]:
                cpu_block_table_lens.append(
                    (seq_id, self.get_seq_num_blocks_allocated(seq_id, BlockDevice.CPU))
                )

        return "\n".join(
            [
                f"Num free GPU blocks: {self.get_num_free_blocks(BlockDevice.GPU)}",
                f"Num free CPU blocks: {self.get_num_free_blocks(BlockDevice.CPU)}",
                f"All GPU block table lens: {gpu_block_table_lens}",
                f"All CPU block table lens: {cpu_block_table_lens}",
            ]
        )


class DryRunBlockSpaceManager(BaseBlockSpaceManager):
    def _init_allocator(
        self, num_blocks: int, watermark: Optional[float] = None
    ) -> DryRunBlockAllocator:
        return DryRunBlockAllocator(num_blocks, watermark=watermark)

    def _free_device_blocks(
        self, seq_id: str, device: BlockDevice, num_blocks_to_free: Optional[int] = None
    ) -> None:
        assert isinstance(seq_id, str)
        assert seq_id in self._block_tables
        assert device in self._block_tables[seq_id]

        num_allocated_blocks = self.get_seq_num_blocks_allocated(seq_id, device)
        if num_blocks_to_free is None:
            num_blocks_to_free = self.get_seq_num_blocks_allocated(seq_id, device)
        assert num_blocks_to_free <= num_allocated_blocks
        self._allocators[device].free(num_blocks_to_free)
        if num_allocated_blocks == num_blocks_to_free:
            del self._block_tables[seq_id][device]
        else:
            self._block_tables[seq_id][device] -= num_blocks_to_free

    def _update_swap_in_mapping(self, seq_id: str, num_blocks_to_swap: int) -> None:
        pass

    def _update_swap_out_mapping(self, seq_id: str, num_blocks_to_swap: int) -> None:
        pass


class BlockSpaceManager(BaseBlockSpaceManager):
    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        duplicate_kv_cache: bool,
        max_model_len: int,
        watermark: float = 0.01,
    ) -> None:
        super().__init__(
            block_size,
            num_gpu_blocks,
            num_cpu_blocks,
            duplicate_kv_cache,
            max_model_len,
            watermark=watermark,
        )
        self.__swap_in_mapping: Dict[str, List[Tuple[int, int]]] = {}
        self.__swap_out_mapping: Dict[str, List[Tuple[int, int]]] = {}

    def _init_allocator(
        self, num_blocks: int, watermark: Optional[float] = None
    ) -> BlockAllocator:
        return BlockAllocator(num_blocks, watermark=watermark)

    def _update_swap_in_mapping(self, seq_id: str, num_blocks_to_swap: int) -> None:
        # If duplicating KV cache, swap in should be a prefix of the CPU/GPU blocks
        # If not duplicating KV cache, length of CPU blocks should be equal to the number of blocks to swap exactly
        if not self.duplicate_kv_cache:
            assert (
                len(self._block_tables[seq_id][BlockDevice.CPU]) == num_blocks_to_swap
            ), f"CPU blocks: {len(self._block_tables[seq_id][BlockDevice.CPU])}, num_blocks_to_swap: {num_blocks_to_swap}"
        self.__swap_in_mapping[seq_id] = list(
            zip(
                self._block_tables[seq_id][BlockDevice.CPU][:num_blocks_to_swap],
                self._block_tables[seq_id][BlockDevice.GPU][:num_blocks_to_swap],
            )
        )

    def _update_swap_out_mapping(self, seq_id: str, num_blocks_to_swap: int) -> None:
        self.__swap_out_mapping[seq_id] = list(
            zip(
                # Prefix of GPU, copy to suffix of CPU
                self._block_tables[seq_id][BlockDevice.GPU][:num_blocks_to_swap],
                self._block_tables[seq_id][BlockDevice.CPU][-num_blocks_to_swap:],
            )
        )

    def get_swap_in_mapping(self, seq_id: str) -> List[Tuple[int, int]]:
        assert isinstance(seq_id, str)
        return self.__swap_in_mapping[seq_id]

    def get_swap_out_mapping(self, seq_id: str) -> List[Tuple[int, int]]:
        assert isinstance(seq_id, str)
        return self.__swap_out_mapping[seq_id]

    def _free_device_blocks(
        self, seq_id: str, device: BlockDevice, num_blocks_to_free: Optional[int] = None
    ) -> None:
        assert isinstance(seq_id, str)
        assert seq_id in self._block_tables
        assert device in self._block_tables[seq_id]

        num_allocated_blocks = self.get_seq_num_blocks_allocated(seq_id, device)
        if num_blocks_to_free is None:
            num_blocks_to_free = self.get_seq_num_blocks_allocated(seq_id, device)
        assert num_blocks_to_free <= num_allocated_blocks
        self._allocators[device].free(
            self._block_tables[seq_id][device][:num_blocks_to_free]
        )  # NOTE: Freeing prefix
        if num_allocated_blocks == num_blocks_to_free:
            del self._block_tables[seq_id][device]
        else:
            self._block_tables[seq_id][device] = self._block_tables[seq_id][device][
                num_blocks_to_free:
            ]  # NOTE: Removing prefix

    def _free_block_table(self, seq_id: str) -> None:
        super()._free_block_table(seq_id)
        if seq_id in self.__swap_in_mapping:
            del self.__swap_in_mapping[seq_id]
        if seq_id in self.__swap_out_mapping:
            del self.__swap_out_mapping[seq_id]

    ########## Metadata ##########

    def get_gpu_block_table(self, seq_id: str) -> List[int]:
        assert isinstance(seq_id, str) and seq_id in self._block_tables
        if BlockDevice.GPU not in self._block_tables[seq_id]:
            return []
        return self._block_tables[seq_id][BlockDevice.GPU]

    def _get_single_duplicate_mapping(
        self, seq: Sequence, num_blocks: int
    ) -> Iterator[Tuple[int, int]]:
        assert self.duplicate_kv_cache
        assert isinstance(seq, Sequence) and self.can_append_slot(seq)
        assert (
            num_blocks <= self.get_seq_num_blocks_allocated(seq.seq_id, BlockDevice.GPU)
        ), f"Requested: {num_blocks}, allocated: {self.get_seq_num_blocks_allocated(seq.seq_id, BlockDevice.GPU)}"
        assert (
            num_blocks <= self.get_seq_num_blocks_allocated(seq.seq_id, BlockDevice.CPU)
        ), f"Requested: {num_blocks}, allocated: {self.get_seq_num_blocks_allocated(seq.seq_id, BlockDevice.CPU)}"
        return list(
            zip(
                self._block_tables[seq.seq_id][BlockDevice.GPU][-num_blocks:],
                self._block_tables[seq.seq_id][BlockDevice.CPU][-num_blocks:],
            )
        )

    def get_duplicate_mapping(
        self, sequences: List[Sequence], seq_num_tokens: List[int]
    ) -> List[Tuple[int, int]]:
        assert len(sequences) == len(seq_num_tokens)
        result = []
        for seq, num_tokens in zip(sequences, seq_num_tokens):
            result.extend(
                self._get_single_duplicate_mapping(
                    seq, (num_tokens + self.block_size - 1) // self.block_size
                )
            )
        return result
