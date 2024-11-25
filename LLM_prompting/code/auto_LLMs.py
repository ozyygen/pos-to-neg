from pathlib import Path
import pandas as pd
from tqdm import tqdm
from huggingface_hub import login
from transformers import pipeline, AutoTokenizer
import torch

def auto_LLMs(rule_path: str, models: list, hf_token: str, output_path: str):
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

    login(token=hf_token) # login to HuggingFace
    torch.manual_seed(0) # set seed

    # Default prompt specifying context, question and instructions to the LLM
    default_prompt = """
        Answer the question based on the context below
        Context: {context}
        Question: {question}
        Answer:
    """

    # Repeat per model
    for ch_model in models:
        tokenizer = AutoTokenizer.from_pretrained(ch_model) # load tokenizer

        # Define the process and modules
        generator = pipeline("text-generation",
                              model=ch_model,
                              tokenizer=tokenizer,
                              torch_dtype=torch.bfloat16,
                              device_map="auto",
                              )
        
        answers = [] # to store model outputs

        # Iterate through prompts
        for ix in tqdm(range(rule.shape[0])):
            prompt = rule["Prompt"].iloc[ix] # get the prompt
            context, question = prompt.split("\n\n", 1) # split prompt's context and question
            input_prompt = default_prompt.format(context=context, question=question) # format prompt template

            # Define generation parameters and get answer from the LLM
            sequences = generator(input_prompt,
                                  max_new_tokens=5, # limits output length significantly
                                  pad_token_id=generator.tokenizer.eos_token_id,
                                  do_sample=True,
                                  top_k=5,
                                  return_full_text=False, # will not return the prompt part
                                  )

            for seq in sequences:
                answers.append(seq["generated_text"]) # append only the string part of the answer object
        
        rule[ch_model] = answers # add answers as a column

        rule.to_csv(output_path / f"{rule_path.stem}.csv", index=None) # write results after each model is done
    
    rule.drop(["Prompt"], axis=1) # no need to keep the prompts
    rule.to_csv(output_path / f"{rule_path.stem}.csv", index=None) # write results after each model is done

# Example

"""
rule_path = "/home/jovyan/work/persistent/LLM_prompting/data/prompts/German.csv"
models = ["Qwen/Qwen2.5-1.5B-Instruct", "facebook/opt-1.3b", "bigscience/bloomz-560m", "openai-community/gpt2"]
hf_token = "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"
output_path = "/home/jovyan/work/persistent/LLM_prompting/data/answers"
auto_LLMs(rule_path=rule_path, models=models, hf_token=hf_token, output_path=output_path)
"""