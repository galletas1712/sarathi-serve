from sarathi.config import SystemConfig
from sarathi.engine.base_llm_engine import BaseLLMEngine


class LLMEngine:

    @classmethod
    def from_system_config(cls, config: SystemConfig) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine = BaseLLMEngine(config)
        # TODO: pipeline parallel llm engine

        return engine
