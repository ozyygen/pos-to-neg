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
    
    tables = [] # to store tables for merging later
    for chosen_rule in chosen_rules: # iterate through all rules
        rule_path = prompts_path / f"{chosen_rule}.csv"
        rule = pd.read_csv(rule_path) # get the table which contains prompts for this rule
        prompts = rule["Prompt"].to_list() # select only the prompts
        
        # Generate per model
        for model in models:
            answers = [] # to store final answers
            new_prompts = [] # to store all answers

            for prompt in tqdm(prompts): # iterate through each prompt for this rule
                sep_prompts = prompt.split("Model's Response:") # split each stage by the specific answer marker
                sep_prompts = list(filter(None, sep_prompts)) # remove empty splits
                
                sub_answers = [] # for model answers per prompt
                for ix in range(len(sep_prompts)):
                    question = sep_prompts[ix] # get the current prompt
                    history = "" # set a blank history

                    if ix != 0:
                        for i in range(ix):
                            history += f"{sep_prompts[i]}\n\nResponse to Step {i+1}: {sub_answers[i]}\n\n"

                    # Formatted instructions
                    messages = [
                        {"role": "system", "content": f"Answer the question based on the provided chat history\nChat history: {history}"},
                        {"role": "user", "content": f"Question: {question}"}
                    ]

                    # Get response
                    if ix == len(sep_prompts)-1:
                        response = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            max_tokens=2 # limit if it is the final Yes/No question
                        )
                    else:
                        response = client.chat.completions.create(
                            model=model,
                            messages=messages
                        )
                        
                    sub_answers.append(response.choices[0].message.content) # store the answers for each reasoning prompt
                
                answers.append(sub_answers[-1]) # store the final answer
                
                # Store prompt answers
                pr = ""
                for i in range(len(sep_prompts)):
                    pr += f"{sep_prompts[i]}\n\nResponse to Step {i+1}: {sub_answers[i]}\n\n"
                new_prompts.append(pr)
                
            rule[f"{model}_Answer"] = answers # add answers as a column
            rule[f"{model}_Responses"] = new_prompts # add all responses as a column
            
            
        rule.to_csv(answers_path / f"{rule_path.stem}.csv", index=None) # write results after all responses are collected for the rule
        print(f"Generated LLM answers for rule {chosen_rule}.\n -----") if verbose else None
        columns_to_remove = ['Prompt'] + [col for col in rule.columns if col.endswith('_Responses')]
        rule = rule.drop(columns=columns_to_remove, errors='ignore')  # Drop specified columns

        tables.append(rule) # store entire table
    final = pd.concat(tables, ignore_index=True) # merge all tables vertically
    final.to_csv(output_path / "answers/chain/final.csv", index=None) # write the final table
    print("Generated final table.") if verbose else None

# Example
prompt_path = "/app/LLM_prompting/data/prompt/Prompt-chaining.txt"
data_path = "/app/LLM_prompting/data/sibling_output.txt"
models = ["qwen2.5-coder:32b","llama3.2","mistral","llama3.1:70b","llama3.1:8b"]
output_path = "/app/LLM_prompting/data"
chosen_rules = ["voice"]
chain_process(prompt_path=prompt_path, data_path=data_path, models=models, output_path=output_path, chosen_rules=chosen_rules)