"""Sequence and its related classes."""

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional

from sarathi.core.datatypes.sampling_params import SamplingParams
from sarathi.core.datatypes.sequence_status import SequenceStatus


@dataclass(frozen=True)
class SequenceInitParams:
    seq_id: str
    prompt: str
    prompt_token_ids: List[int]
    block_size: int
    eos_token_id: int
    arrival_time: float
    sampling_params: SamplingParams


class SequenceBase:
    """Contains the init (frozen) parameters of the sequence."""

    def __init__(
        self,
        seq_id: str,
        prompt: str,
        prompt_token_ids: List[int],
        block_size: int,
        eos_token_id: int,
        arrival_time: float,
        sampling_params: SamplingParams,
    ):
        self._init_params = SequenceInitParams(
            seq_id=seq_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            block_size=block_size,
            eos_token_id=eos_token_id,
            arrival_time=arrival_time,
            sampling_params=sampling_params,
        )
    
    @property
    def seq_id(self) -> str:
        return self._init_params.seq_id
    
    @property
    def prompt(self) -> str:
        return self._init_params.prompt
    
    @property
    def prompt_token_ids(self) -> List[int]:
        return self._init_params.prompt_token_ids
    
    @property
    def block_size(self) -> int:
        return self._init_params.block_size
    
    @property
    def eos_token_id(self) -> int:
        return self._init_params.eos_token_id
    
    @property
    def arrival_time(self) -> float:
        return self._init_params.arrival_time
    
    @property
    def sampling_params(self) -> SamplingParams:
        return self._init_params.sampling_params
    
    def _append_output_tokens_to_prompt_tokens(self, output_token_ids: List[int]):
        self._init_params = SequenceInitParams(
            seq_id=self.seq_id,
            prompt=self.prompt,
            prompt_token_ids=self.prompt_token_ids + output_token_ids,
            block_size=self.block_size,
            eos_token_id=self.eos_token_id,
            arrival_time=self.arrival_time,
            sampling_params=self.sampling_params,
        )


class Sequence(SequenceBase):
    """Stores the data, status, and block information of a sequence.

    Args:
        seq_id: The ID of the sequence.
        prompt: The prompt of the sequence.
        prompt_token_ids: The token IDs of the prompt.
        block_size: The block size of the sequence. Should be the same as the
            block size used by the block manager and cache engine.
    """

    def __init__(
        self,
        seq_id: str,
        prompt: str,
        prompt_token_ids: List[int],
        block_size: int,
        eos_token_id: int,
        arrival_time: float,
        sampling_params: SamplingParams,
    ) -> None:
        super().__init__(
            seq_id=seq_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            block_size=block_size,
            eos_token_id=eos_token_id,
            arrival_time=arrival_time,
            sampling_params=sampling_params,
        )

        self.__output_token_ids: List[int] = []
        self.__num_prompt_tokens_processed = 0

        # Initialize the logical token blocks with the prompt token ids.
        self.__num_logical_blocks = 0
        self.__num_free_slots_last_block = 0

        self.__status = SequenceStatus.WAITING

        # We need to create the logical blocks for the prompt tokens right away.
        self.__create_logical_blocks_for_tokens(self.get_prompt_len())
    
    #################### Derived properties. Everything returned is a copy/cannot be used to alter the state of the sequence. ####################

    # Lengths

    def get_prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    def get_output_len(self) -> int:
        return len(self.__output_token_ids)
    
    def get_total_len(self) -> int:
        return self.get_prompt_len() + self.get_output_len()
    
    # Prefill

    def get_num_prompt_tokens_processed(self) -> int:
        return self.__num_prompt_tokens_processed
    
    def is_prompt_processing_finished(self) -> bool:
        return self.__num_prompt_tokens_processed == len(self.prompt_token_ids)

    def get_next_prompt_chunk_token_ids(self, chunk_size: int) -> List[int]:
        start = self.get_num_prompt_tokens_processed()
        end = start + chunk_size
        assert end <= len(self.prompt_token_ids), (
            f"End index {end} is greater than the prompt length "
            f"{len(self.prompt_token_ids)}"
        )
        return self.prompt_token_ids[start:end]

    def get_next_prompt_chunk_len(self, chunk_size: int) -> int:
        return min(
            chunk_size, len(self.prompt_token_ids) - self.get_num_prompt_tokens_processed()
        )

    # Logical blocks

    def get_num_logical_blocks(self) -> int:
        return self.__num_logical_blocks
    
    # Token IDs

    def get_output_token_ids(self) -> List[int]:
        return deepcopy(self.__output_token_ids)
    
    def get_all_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.__output_token_ids

    def get_last_token_id(self) -> int:
        if not self.__output_token_ids:
            return self.prompt_token_ids[-1]
        return self.__output_token_ids[-1]
    
    #################### Update operations ####################

    # Private

    def __create_logical_blocks_for_tokens(self, num_tokens_to_add: int) -> None:
        # Fill up the last block first
        if self.__num_free_slots_last_block > 0:
            slots_to_occupy = min(self.__num_free_slots_last_block, num_tokens_to_add)
            self.__num_free_slots_last_block -= slots_to_occupy
            num_tokens_to_add -= slots_to_occupy
        
        assert num_tokens_to_add >= 0
        if num_tokens_to_add == 0:
            return
        
        # Now, create as many blocks as necessary
        assert self.__num_free_slots_last_block == 0
        num_blocks_to_add = (num_tokens_to_add + self.block_size - 1) // self.block_size
        self.__num_logical_blocks += num_blocks_to_add
        if num_tokens_to_add % self.block_size == 0:
            self.__num_free_slots_last_block = 0
        else:
            self.__num_free_slots_last_block = self.block_size - (num_tokens_to_add % self.block_size)

    # Public
     
    def update_prompt_tokens_processed(self, num_tokens: int) -> None:
        assert not self.is_prompt_processing_finished()
        assert num_tokens > 0

        self.__num_prompt_tokens_processed += num_tokens
        assert self.__num_prompt_tokens_processed <= len(self.prompt_token_ids)

    def append_token_id(
        self,
        token_id: int,
    ) -> None:
        assert self.is_prompt_processing_finished()
        self.__output_token_ids.append(token_id)
        self.__create_logical_blocks_for_tokens(1)
    
    def reset_for_recompute(self):
        self.set_status(SequenceStatus.WAITING)
        self.__num_prompt_tokens_processed = 0
        self._append_output_tokens_to_prompt_tokens(self.__output_token_ids)
        self.__output_token_ids = []
        # No need to reset logical blocks here
    
        
    #################### State ####################

    def get_status(self) -> SequenceStatus:
        return deepcopy(self.__status)
    
    def set_status(self, new_status: SequenceStatus) -> None:
        SequenceStatus.check_transition(self.get_status(), new_status)
        self.__status = new_status

    def is_finished(self) -> bool:
        return SequenceStatus.is_finished(self.get_status())

    def is_executing(self) -> bool:
        return SequenceStatus.is_executing(self.get_status())

    def is_waiting(self) -> bool:
        return SequenceStatus.is_waiting(self.get_status())

    def is_paused(self) -> bool:
        return SequenceStatus.is_paused(self.get_status())

    def is_running(self) -> bool:
        return SequenceStatus.is_running(self.get_status())
    
    def is_swapping_in(self) -> bool:
        return SequenceStatus.is_swapping_in(self.get_status())
    
    def is_swapped_out(self) -> bool:
        return SequenceStatus.is_swapped_out(self.get_status())

    def check_stop(self) -> None:
        """Stop the finished sequences."""
        # NOTE: This was a bug since a long time ago - __output_text doesn't get updated in worker
        # for stop_str in self.sampling_params.stop:
        #     if self.__output_text.endswith(stop_str):
        #         # Truncate the output text so that the stop string is
        #         # not included in the output.
        #         self.__output_text = self.__output_text[: -len(stop_str)]
        #         self.set_status(SequenceStatus.FINISHED_STOPPED)
        #         return

        # Check if the sequence has reached max_tokens.
        if self.get_output_len() == self.sampling_params.max_tokens:
            self.set_status(SequenceStatus.FINISHED_LENGTH_CAPPED)
            return

        # Check if the sequence has generated the EOS token.
        if (
            not self.sampling_params.ignore_eos
        ) and self.get_last_token_id() == self.eos_token_id:
            self.set_status(SequenceStatus.FINISHED_STOPPED)
            return

    def __repr__(self) -> str:
        return (
            f"{__class__}(seq_id={self.seq_id}, "
            f"status={self.get_status().name}, "
            f"prompt_len={self.get_prompt_len()}, "
            f"output_len={self.get_output_len()}, "
            f"num_logical_blocks={self.get_num_logical_blocks()}, "
            f"num_free_slots_last_block={self.__num_free_slots_last_block}, "
            f"is_prompt_processing_finished={self.is_prompt_processing_finished()}, "
            f"num_prompt_tokens_processed={self.get_num_prompt_tokens_processed()}, "
        )


class DecodeableSequence(Sequence):

    def __init__(
        self,
        seq_id: str,
        prompt: str,
        prompt_token_ids: List[int],
        block_size: int,
        eos_token_id: int,
        arrival_time: float,
        sampling_params: SamplingParams,
    ) -> None:
        super().__init__(
            seq_id=seq_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            block_size=block_size,
            eos_token_id=eos_token_id,
            arrival_time=arrival_time,
            sampling_params=sampling_params,
        )

        # Properties for decoding in engine_sequence_manager
        ## Used for incremental detokenization
        self.__prefix_offset = 0
        self.__read_offset = 0
        ## Input + output tokens
        self.__tokens_decoded_so_far: Optional[List[str]] = None
        self.__output_text = ""

    @property
    def prefix_offset(self) -> int:
        return self.__prefix_offset
    
    @property
    def read_offset(self) -> int:
        return self.__read_offset
    
    @property
    def tokens_decoded_so_far(self) -> Optional[List[int]]:
        return self.__tokens_decoded_so_far
    
    @property
    def output_text(self) -> str:
        return self.__output_text
    
    def update_decode_state(self, new_tokens: List[int], prefix_offset: int, read_offset: int, new_output_text: str) -> None:
        if self.__tokens_decoded_so_far is None:
            self.__tokens_decoded_so_far = new_tokens
        else:
            self.__tokens_decoded_so_far.extend(new_tokens)
        
        self.__prefix_offset = prefix_offset
        self.__read_offset = read_offset
        self.__output_text += new_output_text
    

@dataclass(frozen=True)
class SequenceScheduleMetadata:
    """Metadata generated by the scheduler for sequence that has been scheduled.
    This is passed to the worker, and the sequence manger is responsible for
    materializing it into a `SequenceExecutionMetadata`.

    Args:
        seq_id: The ID of the request.
        prompt_chunk_len: The size of the prompt chunk.
    """

    seq_id: str
    prompt_chunk_len: int

    @property
    def num_prompt_tokens(self) -> int:
        return self.prompt_chunk_len

    @property
    def is_prompt(self) -> bool:
        return self.prompt_chunk_len > 0

    @property
    def num_output_tokens(self) -> int:
        if self.prompt_chunk_len > 0:
            return 0
        return 1

    @property
    def num_tokens(self) -> int:
        return max(self.prompt_chunk_len, 1)

    @classmethod
    def from_sequence(
        cls,
        seq: Sequence,
        prompt_chunk_len: Optional[int] = None,
    ) -> "SequenceScheduleMetadata":
        """NOTE: prompt_chunk_len = None corresponds to the case of no chunked prefill."""

        if prompt_chunk_len is None:
            if seq.is_prompt_processing_finished():
                prompt_chunk_len = 0
            else:
                prompt_chunk_len = seq.get_prompt_len()

        return cls(seq_id=seq.seq_id, prompt_chunk_len=prompt_chunk_len)

    def __str__(self) -> str:
        return (
            f"SequenceScheduleMetadata(seq_id={self.seq_id}, "
            f"prompt_chunk_len={self.prompt_chunk_len})"
        )

    def __repr__(self) -> str:
        return self.__str__()


@dataclass(frozen=True)
class SequenceExecutionMetadata:
    """Metadata for a sequence. Used to create `SamplerMetadata`.

    Args:
        seq: The sequence object.
        block_table: The block table for the sequence
        prompt_chunk_len: The size of the prompt chunk.
    """

    seq: Sequence
    block_table: List[int]
    prompt_chunk_len: int

    @property
    def num_prompt_tokens(self) -> int:
        return self.prompt_chunk_len

    @property
    def is_prompt(self) -> bool:
        return self.prompt_chunk_len > 0

    @property
    def num_output_tokens(self) -> int:
        if self.prompt_chunk_len > 0:
            return 0
        return 1

    @property
    def num_tokens(self) -> int:
        return max(self.prompt_chunk_len, 1)

    def __str__(self) -> str:
        return (
            f"SequenceExecutionMetadata(seq_id={self.seq.seq_id}, "
            f"prompt_chunk_len={self.prompt_chunk_len})"
        )

    def __repr__(self) -> str:
        return self.__str__()


@dataclass(frozen=True)
class SamplerOutput:
    seq_id: str
    output_token: int


SamplerOutputs = List[SamplerOutput]
