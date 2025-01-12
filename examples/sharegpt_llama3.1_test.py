import argparse
import datetime
import json
from typing import List

from tqdm import tqdm

from sarathi import LLMEngine, RequestOutput, SamplingParams
from sarathi.config import (
    ReplicaConfig,
    SystemConfig,
)

BASE_OUTPUT_DIR = "./offline_inference_output/shareGPT"

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)

output_dir = (
    f"{BASE_OUTPUT_DIR}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
)

system_config = SystemConfig(replica_config=ReplicaConfig(output_dir=output_dir))
system_config.cache_config.num_gpu_blocks = 2048

llm_engine = LLMEngine.from_system_config(system_config)


def generate(
    llm_engine: LLMEngine,
    prompts: List[str],
    sampling_params: SamplingParams,
    enable_profiling: bool = False,
) -> List[RequestOutput]:
    for prompt in prompts:
        llm_engine.add_request(prompt, sampling_params)

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
                print("Prompt:", output.prompt)
                print("Generated text:", output.text)
                pbar.update(1)
        iteration += 1

    if enable_profiling:
        llm_engine.stop_profiling()

    pbar.close()

    llm_engine.pull_worker_metrics()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run ShareGPT trace")
    parser.add_argument(
        "path_to_trace",
        type=str,
        help="Directory containing the ShareGPT trace JSON file.",
    )
    args = parser.parse_args()

    # Read the JSON file
    with open(args.path_to_trace, "r") as file:
        trace_data = json.load(file)

    # Convert lists to strings and collect them into a list
    prompts = [" ".join(prompt_list) for prompt_list in trace_data.values()]
