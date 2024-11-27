from pathlib import Path
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from huggingface_hub import login
from transformers import pipeline, AutoTokenizer
import torch

def auto_LLMs(rule_path: str, models: list, output_path: str):
    """
    For a given list of LLMs and a list of prompts, answers are generated, recorded, and written to a CSV file.
    
    Args:
        Path to the file containing rule prompts (rule_path: str) ,
        A list of LLM models (models: list) ,
        User's HuggingFace Token for authentication (hf_token: str) ,
        Path to the folder where the result will be outputted (output_path: str) .
    
    Returns:
        None .
    """

    # Get rule prompts
    rule_path = Path(rule_path)
    rule = pd.read_csv(rule_path)

    output_path = Path(output_path) # get output folder as path

    torch.manual_seed(0) # set seed

    # Default prompt specifying context, question and instructions to the LLM
    default_messages = [
        {"role": "system", "content": "{context}"},
        {"role": "user", "content": "{question}"}
    ]



    # Generate per model
    for model in models:
        client = OpenAI(
        base_url=f"http://qwen2.5-coder:11434/",
        api_key="ollama"
    )
        answers = [] # to store model's answers

        # Iterate through prompts
        for ix in tqdm(range(rule.shape[0])):
            prompt = rule["Prompt"].iloc[ix] # get the prompt
            context, question = prompt.split("\n\n", 1) # split prompt's context and question
            
            # Formatted 
            messages = [
                {"role": "system", "content": f"Answer only in 'Yes' or 'No'\nContext: {context}"},
                {"role": "user", "content": f"Question: {question}"}
            ]
            
            # Get response
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=5 # limit tokens to get desired Yes/No output
            )
            
            answers.append(response.choices[0].message.content) # store answer
        
        rule[ch_model] = answers # add answers as a column

        rule.to_csv(output_path / f"{rule_path.stem}.csv", index=None) # write results after each model is done
    
    rule.drop(["Prompt"], axis=1) # no need to keep the prompts
    rule.to_csv(output_path / f"{rule_path.stem}.csv", index=None) # write results after each model is done

# Example
"""
rule_path = "/home/jovyan/work/persistent/LLM_prompting/data/prompts/German.csv"
models = ["qwen2.5-coder:0.5b"]
output_path = "/home/jovyan/work/persistent/LLM_prompting/data/answers"
auto_LLMs(rule_path=rule_path, models=models, hf_token=hf_token, output_path=output_path)
"""