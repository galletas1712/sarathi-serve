import argparse
import datetime
import json
from typing import Any, Dict, List

from tqdm import tqdm

from sarathi import LLMEngine, RequestOutput, SamplingParams
from sarathi.config import ReplicaConfig, SystemConfig
from sarathi.config.config import FCFSDisaggEmulationSchedulerConfig
from sarathi.engine.base_llm_engine import BaseLLMEngine


def generate(
    llm_engine: LLMEngine,
    prompt_token_id_lists: List[List[int]],
    sampling_params: SamplingParams,
    enable_profiling: bool = False,
) -> List[RequestOutput]:
    for prompt_token_ids in prompt_token_id_lists:
        llm_engine.add_request(
            prompt=None,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
        )

    num_requests = llm_engine.get_num_unfinished_requests()
    pbar = tqdm(total=num_requests, desc="Processed prompts")

    if enable_profiling:
        llm_engine.start_profiling()

    # Run the engine
    iteration = 0
    while llm_engine.has_unfinished_requests():
        step_outputs = llm_engine.step()
        for output in step_outputs:
            if output.finished:
                print("------------------------------------------------")
                print(
                    "### Prompt: ###",
                    llm_engine.tokenizer.decode(output.prompt_token_ids),
                )
                print()
                print("### Generated text: ###", output.text)
                print("------------------------------------------------")
                pbar.update(1)
        iteration += 1

    if enable_profiling:
        llm_engine.stop_profiling()

    pbar.close()

    llm_engine.pull_worker_metrics()


def process_conversations(
    conversations: Dict[str, List[Dict[str, Any]]],
    limit_num_convos: int,
    limit_len: int,
) -> List[List[Dict[str, Any]]]:
    processed_prompts = []
    for i, conversation in enumerate(conversations.values()):
        if i >= limit_num_convos:
            break
        current_prompt = []
        for message in conversation:
            current_prompt.append(message)
            if message["role"] == "user":
                if len(str(current_prompt)) >= limit_len:
                    break
                processed_prompts.append(current_prompt.copy())
                break  # TODO: remove!
    return processed_prompts


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run ShareGPT trace")
    parser.add_argument(
        "path_to_trace",
        type=str,
        help="Directory containing the ShareGPT trace JSON file.",
    )
    args = parser.parse_args()

    # Generation
    BASE_OUTPUT_DIR = "./offline_inference_output/shareGPT"
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)
    output_dir = (
        f"{BASE_OUTPUT_DIR}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    system_config = SystemConfig(
        replica_config=ReplicaConfig(output_dir=output_dir),
        scheduler_config=FCFSDisaggEmulationSchedulerConfig(uses_chunked_prefill=False),
    )
    llm_engine: BaseLLMEngine = LLMEngine.from_system_config(system_config)

    # Read the JSON file
    with open(args.path_to_trace, "r") as file:
        trace_data = json.load(file)
    # Process the conversation data into prompts
    prompts = process_conversations(trace_data, limit_num_convos=500, limit_len=8000)
    prompt_token_id_lists = [
        llm_engine.tokenizer.apply_chat_template(prompt) for prompt in prompts
    ]
    # prompt_token_id_lists = [
    #     llm_engine.tokenizer.apply_chat_template(
    #         [{"role": "user", "content": "What's the capital of Paris?"}], add_generation_prompt=False
    #     )
    # ]

    generate(
        llm_engine=llm_engine,
        prompt_token_id_lists=prompt_token_id_lists,
        sampling_params=sampling_params,
        enable_profiling=False,
    )
