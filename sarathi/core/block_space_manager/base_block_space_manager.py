"""A block manager that manages token blocks."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from collections import deque
from enum import Enum

from sarathi.core.datatypes.block import PhysicalTokenBlock
from sarathi.core.datatypes.sequence import Sequence

class BlockDevice(Enum):
    GPU = "cuda"
    CPU = "cpu"

class BlockAllocator:
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(
        self,
        block_size: int,
        num_blocks: int,
    ) -> None:
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Initialize the free blocks.
        # NOTE: We use a queue here so swapped in blocks just recycle swapped out blocks
        # Hopefully this means better cache locality
        self.free_blocks: deque[PhysicalTokenBlock] = deque()
        for i in range(num_blocks):
            block = PhysicalTokenBlock(block_number=i, block_size=block_size)
            self.free_blocks.append(block)

    def allocate(self) -> PhysicalTokenBlock:
        if not self.free_blocks:
            raise ValueError("Out of memory! No free blocks are available.")
        block = self.free_blocks.popleft()
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        self.free_blocks.append(block)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)


# Mapping: logical block number -> physical block.
BlockTable = List[PhysicalTokenBlock]


class BaseBlockSpaceManager(ABC):
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        max_model_len: int,
        watermark: float = 0.01,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.max_model_len = max_model_len

        self.watermark = watermark
        assert watermark >= 0.0

        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.allocators = {
            BlockDevice.GPU: BlockAllocator(block_size, num_gpu_blocks),
            BlockDevice.CPU: BlockAllocator(block_size, num_cpu_blocks),
        }
        # Mapping: seq_id -> BlockTable.
        self.block_tables: Dict[str, Dict[str, BlockTable]] = {}

        # Stores pending swaps
        self.swap_in_mapping: List[Tuple[int, int]] = []
        self.swap_out_mapping: List[Tuple[int, int]] = []

    @abstractmethod
    def get_num_initial_blocks(self, seq: Sequence) -> int:
        """Returns the number of blocks to allocate for a request initially."""
        pass

    def can_allocate(self, seq: Sequence, device: BlockDevice = BlockDevice.GPU) -> bool:
        num_required_blocks = self.get_num_initial_blocks(seq)
        num_free_blocks = self.allocators[device].get_num_free_blocks()
        # Use watermark to avoid frequent cache eviction.
        return num_free_blocks - num_required_blocks >= self.watermark_blocks

    def allocate(self, seq: Sequence, initial_device: BlockDevice = BlockDevice.GPU) -> None:
        # Allocate physical blocks (on some initial device, either GPU or CPU)
        # NOTE: Most of the time, this should be on GPU.
        # Allocated new physical token blocks that will store the prompt tokens.

        assert self.can_allocate(seq, initial_device)

        block_table: BlockTable = []
        num_initial_blocks = self.get_num_initial_blocks(seq)
        for _ in range(num_initial_blocks):
            block = self.allocators[initial_device].allocate()
            block_table.append(block)

        self.block_tables[seq.seq_id][initial_device] = block_table

    def can_append_slot(self, device: BlockDevice = BlockDevice.GPU) -> bool:
        assert device == BlockDevice.GPU
        num_free_blocks = self.allocators[device].get_num_free_blocks()
        return num_free_blocks > 0

    def append_slot(self, seq: Sequence, device: BlockDevice = BlockDevice.GPU) -> None:
        """Allocate a physical slot for a new token."""
        assert device == BlockDevice.GPU
        assert self.can_append_slot(device)
        logical_blocks = seq.logical_token_blocks
        block_table = self.block_tables[seq.seq_id][device]

        if len(block_table) < len(logical_blocks):
            # The sequence has a new logical block.
            # Allocate a new physical block.
            block = self.allocators[device].allocate()
            block_table.append(block)

    def _get_physical_blocks(self, seq: Sequence, device: BlockDevice = BlockDevice.GPU) -> BlockTable:
        assert seq.is_executing()
        return self.block_tables[seq.seq_id][device]
    
    def _free_device_blocks(self, seq_id: str, device: BlockDevice) -> None:
        block_table = self.block_tables[seq_id][device]
        for block in set(block_table):
            self.allocators[device].free(block)
        del self.block_tables[seq_id][device]

    def _free_block_table(self, seq_id: str) -> None:
        if seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return
        for device in self.block_tables[seq_id].keys():
            self._free_device_blocks(seq_id, device)

    def free(self, seq: Sequence) -> None:
        self._free_block_table(seq.seq_id)
        del self.block_tables[seq.seq_id]

    def reset(self) -> None:
        for seq_id in self.block_tables.keys():
            self._free_block_table(seq_id)
        self.block_tables.clear()
    
    def get_block_table(self, seq: Sequence) -> List[int]:
        block_table = self.block_tables[seq.seq_id]
        return [block.block_number for block in block_table]  # TODO: make into generator instead?

    def is_allocated_in_gpu(self, seq: Sequence) -> bool:
        return seq.seq_id in self.block_tables and BlockDevice.GPU in self.block_tables[seq.seq_id]
    
    def is_allocated_in_cpu(self, seq: Sequence) -> bool:
        return seq.seq_id in self.block_tables and BlockDevice.CPU in self.block_tables[seq.seq_id]
    
    def can_swap_in(self, seq: Sequence) -> bool:
        assert seq.seq_id in self.block_tables
        assert BlockDevice.GPU not in self.block_tables[seq.seq_id] and BlockDevice.CPU in self.block_tables[seq.seq_id]

        num_free_blocks = self.allocators[BlockDevice.GPU].get_num_free_blocks()
        num_required_blocks = len(self.block_tables[seq.seq_id][BlockDevice.CPU])
        return num_free_blocks - num_required_blocks >= self.watermark_blocks

    def swap_in(self, seq: Sequence) -> None:
        assert self.can_swap_in(seq)

        for cpu_block in self.block_tables[seq.seq_id][BlockDevice.CPU]:
            gpu_block = self.allocators[BlockDevice.GPU].allocate()
            self.swap_in_mapping.append((cpu_block.block_number, gpu_block.block_number))
            self.block_tables[seq.seq_id][BlockDevice.GPU].append(gpu_block)

        self._free_device_blocks(seq.seq_id, BlockDevice.CPU)

    def can_swap_out(self, seq: Sequence) -> bool:
        assert seq.seq_id in self.block_tables
        assert BlockDevice.CPU not in self.block_tables[seq.seq_id] and BlockDevice.GPU in self.block_tables[seq.seq_id]

        num_free_blocks = self.allocators[BlockDevice.CPU].get_num_free_blocks()
        num_required_blocks = len(self.block_tables[seq.seq_id][BlockDevice.GPU])
        return num_free_blocks - num_required_blocks >= 0  # NOTE: We don't use watermark for CPU
    
    def swap_out(self, seq: Sequence) -> None:
        assert self.can_swap_out(seq)

        for gpu_block in self.block_tables[seq.seq_id][BlockDevice.GPU]:
            cpu_block = self.allocators[BlockDevice.CPU].allocate()
            self.swap_out_mapping.append((gpu_block.block_number, cpu_block.block_number))
            self.block_tables[seq.seq_id][BlockDevice.CPU].append(cpu_block)

        self._free_device_blocks(seq.seq_id, BlockDevice.GPU)
    
    def get_and_clear_swap_mappings(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        swap_in_mapping = self.swap_in_mapping
        swap_out_mapping = self.swap_out_mapping
        self.swap_in_mapping = []
        self.swap_out_mapping = []
        return swap_in_mapping, swap_out_mapping