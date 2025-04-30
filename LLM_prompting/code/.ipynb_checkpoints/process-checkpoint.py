from prompt_generator import prompt_generator
from auto_LLMs import auto_LLMs
from pathlib import Path
import os
import pandas as pd

def process(prompt_path: str, data_path: str, models: list, hf_token: str, output_path: str, chosen_rules="all", verbose=False):
    """
    Generates prompts for all rules, gets results from a series of LLMs per chosen rules.

    Args:
        Path to the file containing a generic LLM prompt (prompt_path: str) ,
        Path to the file containing labels and their outputs to be substitutes (data_path: str) ,
        A list of LLM models (models: list) ,
        User's HuggingFace Token for authentication (hf_token: str) ,
        Path to the folder where resulting components will be outputted (output_path: str) ,
        A list of rules whose prompts will be processed by the LLMs, considers all rules by default (chosen_rules: "all" or list) ,
        Indicator for if the progress should be printed, False by default (verbose: bool) .

    Returns:
        None .
    """

    print("Starting process.\n -----") if verbose else None

    # Create the output folder where all resulting components will be stored
    os.makedirs(output_path, exist_ok=True)
    output_path = Path(output_path)

    # Create a prompts folder where formatted prompts per rule will be written
    prompts_path = output_path / "prompts"
    os.makedirs(prompts_path, exist_ok=True)

    # Generate prompts for all rules
    prompt_generator(prompt_path=prompt_path, data_path=data_path, output_path=prompts_path)
    print("Generated prompts.\n -----") if verbose else None

    # Create an answers folder where LLM answers per rule will be written
    answers_path = output_path / "answers"
    os.makedirs(answers_path, exist_ok=True)
    
    # Choose all rules if not specified in input
    if chosen_rules == "all":
        chosen_rules = [ x.stem for x in prompts_path.glob("**/*") if x.is_file() ]
        
    # Get LLMs' results per rule
    for rule in chosen_rules:
        rule_path = prompts_path / f"{rule}.csv"
        auto_LLMs(rule_path=rule_path, models=models, hf_token=hf_token, output_path=answers_path)
        print(f"Generated LLM answers for rule {rule}.\n -----") if verbose else None

    print("Done.") if verbose else None

# Example
prompt_path = "/home/jovyan/work/persistent/LLM_prompting/data/input/LLM-prompts.txt"
data_path = "/home/jovyan/work/persistent/LLM_prompting/data/input/output-with-labels.txt"
models = ["Qwen/Qwen2.5-1.5B-Instruct", "facebook/opt-1.3b", "bigscience/bloomz-560m", "openai-community/gpt2"]
hf_token = "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"
output_path = "/home/jovyan/work/persistent/LLM_prompting/data"
chosen_rules = ["English", "voice", "Spanish", "film_genre"]
process(prompt_path=prompt_path, data_path=data_path, models=models, hf_token=hf_token, output_path=output_path, chosen_rules=chosen_rules)