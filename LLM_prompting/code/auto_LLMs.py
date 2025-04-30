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
  # Load the rules
    rule_path = Path(rule_path)
    rule = pd.read_csv(rule_path)

    # Ensure output path exists
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    client = OpenAI(
    base_url="http://ollama:11434/v1/",  # Ensure this URL is correct
    api_key="ollama"  # Replace with the actual API key
)   

    torch.manual_seed(0)
    for model in models:
        print(f"Processing model: {model}")
        answers = []  # To store model's answers

        # Iterate through prompts
        for ix in tqdm(range(rule.shape[0]), desc=f"Evaluating prompts for {model}"):
            try:
                
                # Get the prompt and clean unnecessary spaces
                prompt = rule["Prompt"].iloc[ix].strip()
                context, question = prompt.split("\n\n", 1)  # Split prompt into context and question
                
                # Format messages
                messages = [
                    {"role": "system", "content": f"Answer only in 'Yes' or 'No'\nContext: {context}"},
                    {"role": "user", "content": f"Question: {question}"}
                ]

                # Get response from the model
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=5  # Limit tokens to get concise Yes/No output
                )

                # Extract and store the answer
                answer = response.choices[0].message.content.strip()

                answers.append(answer)

            except Exception as e:
                print(f"Error for prompt {ix} with model {model}: {e}")
                answers.append("Error")

        # Add answers as a column to the DataFrame
        rule[model] = answers

        # Save intermediate results for the current model
        rule.to_csv(output_path / f"{rule_path.stem}_{model}.csv", index=False)
        print(f"Results saved for model {model} to {output_path / f'{rule_path.stem}_{model}.csv'}")

    # Remove the "Prompt" column as it is no longer needed
    rule.drop(columns=["Prompt"], inplace=True)

    # Save the final results
    rule.to_csv(output_path / f"{rule_path.stem}_final.csv", index=False)
    print(f"Final results saved to {output_path / f'{rule_path.stem}_final.csv'}")


# Example
"""
rule_path = "/home/jovyan/work/persistent/LLM_prompting/data/prompts/German.csv"
models = ["qwen2.5-coder:0.5b"]
output_path = "/home/jovyan/work/persistent/LLM_prompting/data/answers"
auto_LLMs(rule_path=rule_path, models=models, hf_token=hf_token, output_path=output_path)
"""