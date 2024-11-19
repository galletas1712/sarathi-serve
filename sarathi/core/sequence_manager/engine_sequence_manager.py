from typing import Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from sarathi.config import SystemConfig
from sarathi.core.datatypes.sequence import DecodeableSequence
from sarathi.core.sequence_manager.base_sequence_manager import BaseSequenceManager
from sarathi.utils.transformers.tokenizer import detokenize_incrementally


class EngineSequenceManager(BaseSequenceManager):

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        config: SystemConfig,
    ):
        super().__init__(config)
        self.tokenizer = tokenizer

    def _on_append_token(self, seq: DecodeableSequence) -> None:
        """When a token is appended, the engine will decode the new token for a sequence."""

        (new_tokens, new_output_text, prefix_offset, read_offset) = (
            detokenize_incrementally(
                self.tokenizer,
                all_input_ids=seq.get_all_token_ids(),
                prev_tokens=seq.tokens_decoded_so_far,
                prefix_offset=seq.prefix_offset,
                read_offset=seq.read_offset,
                skip_special_tokens=True,
            )
        )
        seq.update_decode_state(
            new_tokens=new_tokens,
            prefix_offset=prefix_offset,
            read_offset=read_offset,
            new_output_text=new_output_text,
        )