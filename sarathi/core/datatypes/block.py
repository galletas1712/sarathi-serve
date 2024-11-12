"""Token blocks."""

from typing import List

class PhysicalTokenBlock:
    """Represents the state of a block in the KV cache."""

    def __init__(
        self,
        block_number: int,
        block_size: int,
    ) -> None:
        self.block_number = block_number
        self.block_size = block_size

    def __repr__(self) -> str:
        return str(self.block_number)