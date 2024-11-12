"""A block manager that manages token blocks."""

from typing import Dict, List, Optional, Tuple
from enum import Enum

from sarathi.core.datatypes.sequence import Sequence


# Mapping: logical block number -> physical block number.
BlockNumber = int
BlockTable = List[BlockNumber]


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
        num_blocks: int,
    ) -> None:
        self.__num_blocks = num_blocks

        # Initialize the free blocks.
        self.__free_blocks: List[BlockNumber] = []
        for i in reversed(range(num_blocks)):
            self.__free_blocks.append(i)
    
    @property
    def num_total_blocks(self) -> int:
        return self.__num_blocks

    def allocate(self) -> BlockNumber:
        if not self.__free_blocks:
            raise ValueError("Out of memory! No free blocks are available.")
        block = self.__free_blocks.pop()
        return block

    def free(self, block: BlockNumber) -> None:
        self.__free_blocks.append(block)

    def get_num_free_blocks(self) -> int:
        return len(self.__free_blocks)


class BlockSpaceManager:
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
            BlockDevice.GPU: BlockAllocator(num_gpu_blocks),
            BlockDevice.CPU: BlockAllocator(num_cpu_blocks),
        }
        # Mapping: seq_id -> BlockTable.
        self.block_tables: Dict[str, Dict[str, BlockTable]] = {}

        # Stores pending swaps
        self.swap_in_mapping: Dict[str, List[Tuple[int, int]]] = {}
        self.swap_out_mapping: Dict[str, List[Tuple[int, int]]] = {}

    def get_num_initial_blocks(self, seq: Sequence) -> int:
        return seq.get_num_logical_blocks()

    def _ensure_valid(self) -> None:
        for seq_id in self.block_tables.keys():
            assert self.num_blocks_allocated(seq_id, BlockDevice.GPU) == len(self.get_gpu_block_table(seq_id))
            assert self.num_blocks_allocated(seq_id, BlockDevice.CPU) == len(self.get_cpu_block_table(seq_id))

        for device in [BlockDevice.GPU, BlockDevice.CPU]:
            total_blocks_in_block_table = 0
            for seq_id in self.block_tables.keys():
                total_blocks_in_block_table += self.num_blocks_allocated(seq_id, device)
            
            # print(f"Total blocks in block table: {total_blocks_in_block_table}, total blocks: {self.allocators[device].num_blocks}, num free blocks: {self.allocators[device].get_num_free_blocks()}")
            assert total_blocks_in_block_table == self.allocators[device].num_total_blocks - self.allocators[device].get_num_free_blocks()
        
    def can_allocate(self, seq: Sequence, device: BlockDevice = BlockDevice.GPU) -> bool:
        assert isinstance(seq, Sequence)
        num_required_blocks = self.get_num_initial_blocks(seq)
        num_free_blocks = self.allocators[device].get_num_free_blocks()
        # Use watermark to avoid frequent cache eviction.
        return num_free_blocks - num_required_blocks >= self.watermark_blocks

    def allocate(self, seq: Sequence, initial_device: BlockDevice = BlockDevice.GPU) -> None:
        assert isinstance(seq, Sequence)
        # Allocate physical blocks (on some initial device, either GPU or CPU)
        # NOTE: Most of the time, this should be on GPU.
        # Allocated new physical token blocks that will store the prompt tokens.

        assert self.can_allocate(seq, initial_device)

        block_table: BlockTable = []
        num_initial_blocks = self.get_num_initial_blocks(seq)
        for _ in range(num_initial_blocks):
            block = self.allocators[initial_device].allocate()
            block_table.append(block)

        self.block_tables[seq.seq_id] = {initial_device: block_table}
        self._ensure_valid()
    
    def num_blocks_allocated(self, seq_id: str, device: BlockDevice) -> int:
        assert isinstance(seq_id, str)
        if device not in self.block_tables[seq_id]:
            return 0
        return len(self.block_tables[seq_id][device])
    
    def num_blocks_remaining_after(self, num_required_blocks: int, device: BlockDevice = BlockDevice.GPU, use_watermark: bool = True) -> int:
        num_free_blocks = self.allocators[device].get_num_free_blocks()
        return num_free_blocks - (self.watermark_blocks if use_watermark else 0) - num_required_blocks
    
    def can_append_slot(self, seq: Sequence, device: BlockDevice = BlockDevice.GPU) -> bool:
        assert isinstance(seq, Sequence)
        assert device == BlockDevice.GPU
        block_table_len = self.num_blocks_allocated(seq.seq_id, device)
        assert seq.get_num_logical_blocks() - block_table_len <= 1
        return self.num_blocks_remaining_after(
            num_required_blocks=1 if block_table_len < seq.get_num_logical_blocks() else 0,
            device=device
        ) >= 0
    
    def append_slot(self, seq: Sequence, device: BlockDevice = BlockDevice.GPU) -> None:
        assert isinstance(seq, Sequence)
        """Allocate a physical slot for a new token."""
        assert device == BlockDevice.GPU
        block_table = self.block_tables[seq.seq_id][device]

        if len(block_table) < seq.get_num_logical_blocks():
            # The sequence has a new logical block.
            # Allocate a new physical block.
            assert self.can_append_slot(seq, device)
            block = self.allocators[device].allocate()
            block_table.append(block)
        
        self._ensure_valid()

    def _free_device_blocks(self, seq_id: str, device: BlockDevice) -> None:
        assert isinstance(seq_id, str)
        block_table = self.block_tables[seq_id][device]
        for block in set(block_table):
            self.allocators[device].free(block)
        self.block_tables[seq_id].pop(device)
        self._ensure_valid()

    def _free_block_table(self, seq_id: str) -> None:
        assert isinstance(seq_id, str)
        if seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return
        devices = list(self.block_tables[seq_id].keys())
        for device in devices:
            self._free_device_blocks(seq_id, device)
        self._ensure_valid()

    def free(self, seq_id: str) -> None:
        assert isinstance(seq_id, str)
        self._free_block_table(seq_id)
        self.block_tables.pop(seq_id)
        self._ensure_valid()

    def reset(self) -> None:
        for seq_id in self.block_tables.keys():
            self._free_block_table(seq_id)
        self.block_tables.clear()
        self._ensure_valid()
    
    def get_gpu_block_table(self, seq_id: str) -> List[int]:
        assert isinstance(seq_id, str)
        if BlockDevice.GPU not in self.block_tables[seq_id]:
            return []
        return self.block_tables[seq_id][BlockDevice.GPU]

    def get_cpu_block_table(self, seq_id: str) -> List[int]:
        assert isinstance(seq_id, str)
        if BlockDevice.CPU not in self.block_tables[seq_id]:
            return []
        return self.block_tables[seq_id][BlockDevice.CPU]
    
    def get_block_table_metadata(self) -> List[Tuple[int, int]]:
        result_gpu = []
        result_cpu = []
        for seq_id in self.block_tables.keys():
            if BlockDevice.GPU in self.block_tables[seq_id]:
                result_gpu.append((seq_id, len(self.block_tables[seq_id][BlockDevice.GPU])))
            if BlockDevice.CPU in self.block_tables[seq_id]:
                result_cpu.append((seq_id, len(self.block_tables[seq_id][BlockDevice.CPU])))
        
        return result_gpu, result_cpu

    def is_allocated_in_gpu(self, seq_id: str) -> bool:
        assert isinstance(seq_id, str)
        return seq_id in self.block_tables and BlockDevice.GPU in self.block_tables[seq_id]
    
    def is_allocated_in_cpu(self, seq_id: str) -> bool:
        assert isinstance(seq_id, str)
        return seq_id in self.block_tables and BlockDevice.CPU in self.block_tables[seq_id]
    
    def _can_swap_in(self, seq_id: str, num_logical_blocks: Optional[int]) -> bool:
        # NOTE: If num_logical_blocks is None, this means we use the number of blocks already allocated (not the number of logical blocks required in the decode step)

        assert isinstance(seq_id, str)
        # print(f"Can swap in? {seq_id}, {list(self.block_tables.keys())}")
        assert seq_id in self.block_tables
        assert BlockDevice.GPU not in self.block_tables[seq_id] and BlockDevice.CPU in self.block_tables[seq_id]

        # If we want to append a slot right after swap in (if needed), we measure required blocks using logical token blocks, as in append_slot
        # Otherwise, we only need however many blocks were already allocated
        if num_logical_blocks is None:
            num_logical_blocks = self.num_blocks_allocated(seq_id, BlockDevice.CPU)

        return self.num_blocks_remaining_after(num_logical_blocks, BlockDevice.GPU) >= 0
    
    def can_swap_in(self, seq_id: str) -> bool:
        assert isinstance(seq_id, str)
        return self._can_swap_in(seq_id, None)
    
    def can_swap_in_and_append_slot(self, seq_id: str, num_logical_blocks: int) -> bool:
        assert isinstance(seq_id, str)
        return self._can_swap_in(seq_id, num_logical_blocks)
    
    def begin_swap_in(self, seq_id: str):
        assert isinstance(seq_id, str)
        assert self.can_swap_in(seq_id)

        self.block_tables[seq_id][BlockDevice.GPU] = []
        swap_in_mapping = []

        for cpu_block in self.block_tables[seq_id][BlockDevice.CPU]:
            gpu_block = self.allocators[BlockDevice.GPU].allocate()
            swap_in_mapping.append((cpu_block, gpu_block))
            self.block_tables[seq_id][BlockDevice.GPU].append(gpu_block)
        
        self.swap_in_mapping[seq_id] = swap_in_mapping
        self._ensure_valid()
        # print(f"Begin swap in {seq_id} {list(self.block_tables.keys())}")
        
    def finish_swap_in(self, seq_id: str):
        assert isinstance(seq_id, str)
        self._free_device_blocks(seq_id, BlockDevice.CPU)
        self._ensure_valid()
        # print(f"Finish swap in {seq_id} {list(self.block_tables.keys())}")

    def can_swap_out(self, seq_id: str) -> bool:
        assert isinstance(seq_id, str)
        assert seq_id in self.block_tables
        assert BlockDevice.CPU not in self.block_tables[seq_id] and BlockDevice.GPU in self.block_tables[seq_id]

        num_required_blocks = self.num_blocks_allocated(seq_id, BlockDevice.GPU)
        return self.num_blocks_remaining_after(num_required_blocks, BlockDevice.CPU) >= 0
    
    def swap_out(self, seq_id: str):
        assert isinstance(seq_id, str)
        assert self.can_swap_out(seq_id)

        self.block_tables[seq_id][BlockDevice.CPU] = []
        swap_out_mapping = []

        for gpu_block in self.block_tables[seq_id][BlockDevice.GPU]:
            cpu_block = self.allocators[BlockDevice.CPU].allocate()
            swap_out_mapping.append((gpu_block, cpu_block))
            self.block_tables[seq_id][BlockDevice.CPU].append(cpu_block)
        
        self.swap_out_mapping[seq_id] = swap_out_mapping
        self._free_device_blocks(seq_id, BlockDevice.GPU)
        self._ensure_valid()
    
    def get_swap_in_mapping(self, seq_id: str) -> List[int]:
        assert isinstance(seq_id, str)
        return self.swap_in_mapping[seq_id]
    
    def get_swap_out_mapping(self, seq_id: str) -> List[int]:
        assert isinstance(seq_id, str)
        return self.swap_out_mapping[seq_id]
