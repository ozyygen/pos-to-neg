from prompt_generator import prompt_generator
from pathlib import Path
import os
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

def chain_process(prompt_path: str, data_path: str, models: list, output_path: str, chosen_rules="all", verbose=False):
    """
    Generates prompts for all rules, gets results from a series of LLMs per chosen rules.

    Args:
        Path to the file containing a generic LLM prompt (prompt_path: str) ,
        Path to the file containing labels and their outputs to be substitutes (data_path: str) ,
        A list of LLM models (models: list) ,
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

    client = OpenAI(
        base_url="http://ollama:11434/v1/",
        api_key="ollama"
    )

    # Generate per model
    for model in models:
        for chosen_rule in chosen_rules:  # iterate through all rules
            rule_path = prompts_path / f"{chosen_rule}.csv"
            rule = pd.read_csv(rule_path)  # get the table which contains prompts for this rule
            prompts = rule["Prompt"].to_list()  # select only the prompts
            
            answers = []  # Reset answers for the current rule

            for prompt in tqdm(prompts):  # iterate through each prompt for this rule
                sep_prompts = prompt.split("Model's Response:")  # split each stage by the specific answer marker
                sep_prompts = list(filter(None, sep_prompts))  # remove empty splits

                sub_answers = []  # Store model answers for each step
                for ix in range(len(sep_prompts)):
                    question = sep_prompts[ix]  # get the current prompt
                    history = ""  # set a blank history

                    if ix != 0:
                        for i in range(ix):
                            history += f"{sep_prompts[i]}\n\nResponse to Step {i+1}: {sub_answers[i]}\n\n"

                    # Formatted instructions
                    messages = [
                        {"role": "system", "content": f"Answer the question based on the provided chat history\nChat history: {history}"},
                        {"role": "user", "content": f"Question: {question}"}
                    ]

                    # Get response
                    if ix == len(sep_prompts) - 1:
                        response = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            max_tokens=2  # limit if it is the final Yes/No question
                        )
                    else:
                        response = client.chat.completions.create(
                            model=model,
                            messages=messages
                        )

                    sub_answers.append(response.choices[0].message.content)  # store the answers for each reasoning prompt

                answers.append(sub_answers[-1])  # Store the final answer for this prompt

            # Debugging step: Ensure lengths match
            if len(answers) != len(rule):
                print(f"Error: Length mismatch for {chosen_rule} (answers: {len(answers)}, rows: {len(rule)})")
                continue  # Skip this rule to avoid writing mismatched data

            # Add answers as a new column
            rule[model] = answers
            print(f"Generated LLM answers for rule {chosen_rule}.\n -----") if verbose else None

            # Save updated rule DataFrame
            rule.to_csv(output_path / f"{rule_path.stem}.csv", index=None)

    print("Done.") if verbose else None

# Example
prompt_path = "/app/LLM_prompting/data/prompt/Prompt-chaining.txt"
data_path = "/app/LLM_prompting/data/sibling_output.txt"
models = ["qwen2.5-coder:32b","llama3.2","mistral","llama3.1:70b","llama3.1:8b"]
output_path = "/app/LLM_prompting/data"
chosen_rules = ["English","human","voice"]
chain_process(prompt_path=prompt_path, data_path=data_path, models=models, output_path=output_path, chosen_rules=chosen_rules)