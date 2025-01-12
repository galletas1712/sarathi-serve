from typing import List, Optional, Tuple

import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper, append_paged_kv_cache

from sarathi.config import ModelConfig, ParallelConfig
from sarathi.core.datatypes.sequence import SequenceExecutionMetadata
from sarathi.metrics.constants import OperationMetrics
from sarathi.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper


class FlashinferAttentionWrapper(BaseAttentionWrapper):
    """
    Wraps all attention operations in flashinfer and handles waiting on async swap-ins, performing the duplicate KV swap-out and waiting on async swap-out.
    """

    _inst = None

    def init(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        block_size: int,
        device: torch.device,
    ):
        super().init(model_config, parallel_config, block_size, device)

        prefill_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            prefill_workspace_buffer, "NHD"
        )

        decode_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self.decode_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            decode_workspace_buffer, "NHD"
        )

        self.is_metadata_initialized = False
        self.is_profiling_iteration = False
        self.contains_prefill = False
        self.contains_decode = False
        self.num_prefill_tokens = 0
        self.num_total_tokens = 0

        self.append_qo_indptr_tensor = None
        self.append_kv_page_indices_tensor = None
        self.append_kv_page_indptr_tensor = None
        self.append_kv_last_page_len_tensor = None

    def to_int_tensor(self, data: List[int]) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.int32, device="cuda")

    def begin_forward(
        self,
        seq_exec_metadata_list: List[SequenceExecutionMetadata],
        duplicate_mapping: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        # The indptr tensor captures the location query tokens in the input tensor.
        # |<---------------------- num_valid_tokens ----------------------------------------------------->|
        # |<--------------- num_prompt_tokens -------------->||<------- num_generation_tokens (M) ------->|
        # |<--prompt_0-->|<--prompt_1-->|...|<--prompt_N-1-->||<--generation_0-->|...|<--generation_M-1-->|<--padding-->|
        #
        # Flashinfer calls this layout as a raggedtensor. The indptr tensor captures the start of each
        # sequence in the ragged tensor. The length of the indptr tensor is the number of sequences + 1.
        # We perform both prefill and decode attention in a single call to batched prefill kernel.
        # prefill_qo_indptr: [0, prompt_0, prompt_0 + prompt_1, ..., prompt_0 + ... + prompt_N-1, generation_0, generation_0 + 1, ..., generation_0 + ... + M]
        prefill_qo_indptr: List[int] = [0]
        decode_qo_indptr: List[int] = [0]
        # The kv_page_indices tensor captures the pages of the key-value cache that
        # are assigned to each token in the input tensor. Since there is a variable number
        # of pages assigned to each sequence, a ragged tensor to represent this.
        prefill_kv_page_indices: List[int] = []
        decode_kv_page_indices: List[int] = []
        # the last page might not be full, so we need to keep track of the length of the last page
        prefill_kv_last_page_len: List[int] = []
        decode_kv_last_page_len: List[int] = []
        # Since the prefill_kv_page_indices tensor is a ragged tensor, we also need to keep track of the
        # indptr tensor for the prefill_kv_page_indices tensor. This tensor captures the start of each sequence
        # in the ragged tensor.
        prefill_kv_page_indptr: List[int] = [0]
        decode_kv_page_indptr: List[int] = [0]

        self.is_profiling_iteration = False
        self.is_metadata_initialized = True

        self.contains_prefill = False
        self.contains_decode = False

        for seq_exec_metadata in seq_exec_metadata_list:
            if not seq_exec_metadata.is_prompt:
                continue

            # ONLY used for profiling
            if seq_exec_metadata.block_table is None:
                self.is_profiling_iteration = True
                # During memory profiling, the block tables are not initialized yet.
                #  We will just skip the attention computation for now.
                return

            self.contains_prefill = True

            prompt_chunk_len = seq_exec_metadata.prompt_chunk_len
            processed_prompt_len = (
                seq_exec_metadata.seq.get_num_prompt_tokens_processed()
            )
            current_total_len = processed_prompt_len + prompt_chunk_len

            # indptr for the prompt tokens in q/o tensor
            prefill_qo_indptr.append(prefill_qo_indptr[-1] + prompt_chunk_len)
            # Compute the kv page indices for the prompt tokens.
            num_blocks_in_use = (
                current_total_len + self.block_size - 1
            ) // self.block_size

            # NOTE: We don't assert the block table to be equal to number of blocks in use in prefill
            # because we allocated the full sequence at the start.

            prefill_kv_page_indices.extend(
                seq_exec_metadata.block_table[:num_blocks_in_use]
            )
            prefill_kv_page_indptr.append(
                prefill_kv_page_indptr[-1] + num_blocks_in_use
            )
            prefill_kv_last_page_len.append(
                current_total_len % self.block_size or self.block_size
            )

        for seq_exec_metadata in seq_exec_metadata_list:
            if seq_exec_metadata.is_prompt:
                continue

            if seq_exec_metadata.block_table is None:
                self.is_profiling_iteration = True
                return

            self.contains_decode = True

            context_len = seq_exec_metadata.seq.get_total_len()
            # indptr for the prompt tokens in q/o tensor
            decode_qo_indptr.append(decode_qo_indptr[-1] + 1)
            # Compute the kv page indices for the prompt tokens.
            num_blocks_in_use = (context_len + self.block_size - 1) // self.block_size

            assert (
                num_blocks_in_use == seq_exec_metadata.seq.get_num_logical_blocks()
            ), (
                f"Number of blocks in use {num_blocks_in_use} does not match the number of logical blocks "
                f"{seq_exec_metadata.seq.get_num_logical_blocks()}"
            )
            assert num_blocks_in_use == len(seq_exec_metadata.block_table), (
                f"Number of blocks in use {num_blocks_in_use} does not match the length of the block table "
                f"{len(seq_exec_metadata.block_table)}"
            )

            decode_kv_page_indices.extend(
                seq_exec_metadata.block_table[:num_blocks_in_use]
            )
            decode_kv_page_indptr.append(decode_kv_page_indptr[-1] + num_blocks_in_use)
            decode_kv_last_page_len.append(
                context_len % self.block_size or self.block_size
            )

        if self.contains_prefill:
            self.prefill_wrapper.begin_forward(
                self.to_int_tensor(prefill_qo_indptr),
                self.to_int_tensor(prefill_kv_page_indptr),
                self.to_int_tensor(prefill_kv_page_indices),
                self.to_int_tensor(prefill_kv_last_page_len),
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
            )

        if self.contains_decode:
            self.decode_wrapper.begin_forward(
                self.to_int_tensor(decode_qo_indptr),
                self.to_int_tensor(decode_kv_page_indptr),
                self.to_int_tensor(decode_kv_page_indices),
                self.to_int_tensor(decode_kv_last_page_len),
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
            )

        self.num_prefill_tokens = prefill_qo_indptr[-1]
        self.num_total_tokens = self.num_prefill_tokens + len(decode_qo_indptr) - 1

        self.append_qo_indptr_tensor = self.to_int_tensor(
            prefill_qo_indptr[:-1]
            + [x + prefill_qo_indptr[-1] for x in decode_qo_indptr]
        )
        self.append_kv_page_indices_tensor = self.to_int_tensor(
            prefill_kv_page_indices + decode_kv_page_indices
        )
        self.append_kv_page_indptr_tensor = self.to_int_tensor(
            prefill_kv_page_indptr[:-1]
            + [x + prefill_kv_page_indptr[-1] for x in decode_kv_page_indptr]
        )
        self.append_kv_last_page_len_tensor = self.to_int_tensor(
            prefill_kv_last_page_len + decode_kv_last_page_len
        )

        # NOTE: Need to make sure it's initialized first
        # Only time it's not initialized is when profiling available blocks
        if self.cache_engine is not None:
            self.cache_engine.register_duplicate_mapping_optional(duplicate_mapping)

    def end_forward(self):
        if self.contains_prefill:
            self.prefill_wrapper.end_forward()

        if self.contains_decode:
            self.decode_wrapper.end_forward()

        self.is_metadata_initialized = False

        # NOTE: Synchronize the duplication stream at the end of ALL layers
        if self.cache_engine is not None:
            self.cache_engine.sync_and_remove_duplicate_mapping_optional()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        global_layer_id: int,
        worker_layer_id: int,
        softmax_scale: float = 1.0,
    ) -> torch.Tensor:
        assert self.is_metadata_initialized, "Metadata is not initialized."

        if self.is_profiling_iteration:
            # there is no need to call attention in profiling mode
            return torch.zeros_like(query)

        with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, global_layer_id):
            query = query.contiguous().reshape(-1, self.num_q_heads, self.head_dim)
            key = key.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)
            value = value.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)

        output = torch.empty_like(query)
        if self.cache_engine is not None and self.cache_engine.async_swap_in:
            self.cache_engine.wait_for_swap_in(layer=worker_layer_id)

        with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, global_layer_id):
            append_paged_kv_cache(
                key,
                value,
                self.append_qo_indptr_tensor,
                self.cache_engine.gpu_cache[worker_layer_id]
                if self.cache_engine is not None
                else [None],
                self.append_kv_page_indices_tensor,
                self.append_kv_page_indptr_tensor,
                self.append_kv_last_page_len_tensor,
                kv_layout="NHD",
            )

        # TODO: timer
        with self.get_timer(OperationMetrics.ATTN_PREFILL, global_layer_id):
            if self.contains_prefill:
                output[: self.num_prefill_tokens] = self.prefill_wrapper.forward(
                    query[: self.num_prefill_tokens],
                    self.cache_engine.gpu_cache[worker_layer_id]
                    if self.cache_engine is not None
                    else [None],
                    pos_encoding_mode="NONE",
                    sm_scale=softmax_scale,
                )

        with self.get_timer(OperationMetrics.ATTN_DECODE, global_layer_id):
            if self.contains_decode:
                output[self.num_prefill_tokens : self.num_total_tokens] = (
                    self.decode_wrapper.forward(
                        query[self.num_prefill_tokens : self.num_total_tokens],
                        self.cache_engine.gpu_cache[worker_layer_id]
                        if self.cache_engine is not None
                        else [None],
                        pos_encoding_mode="NONE",
                        sm_scale=softmax_scale,
                    )
                )

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, global_layer_id):
            output = output.reshape(-1, self.num_q_heads * self.head_dim)

        if self.cache_engine is not None:
            self.cache_engine.swap_out_duplicate_kv_cache(worker_layer_id)

        return output
