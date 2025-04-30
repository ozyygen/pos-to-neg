from pathlib import Path
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import re

def evaluate_rules_with_llms(rule_path: str, models: list, output_csv_path: str):
    """
    Evaluates each rule in a CSV file using multiple LLM models by asking whether the rule is clinically relevant.

    """
    # Load the rules
    rule_path = Path(rule_path)
    df = pd.read_csv(rule_path)
    
    # Prepare AI client 
    client = OpenAI(
        base_url="path", 
        api_key="ollama"
    )
    
    # Collect answers in a new list of dicts
    output_data = []

    for model in models:
        print(f"Evaluating with model: {model}")
        for ix in tqdm(range(df.shape[0]), desc=f"Processing model {model}"):
            try:
                rule_text = df["Rule_With_Descriptions"].iloc[ix].strip()
                prompt = f"Horn rules are logical expressions consisting of a rule body (premises) and a rule head (conclusion), structured as implications. They follow the form: rel1(A,B) and rel3(B,C) => rel2(A,C). The left side of the arrow (antecedent) represents the rule body, containing one or more conditions that must be satisfied. The right side of the arrow (consequent) is the rule head, which follows if the body conditions hold. Each element follows the structure relation(subject, object), where relations define connections between entities. Is this rule '{rule_text}' clinically relevant? Answer only yes or no."
                
                messages = [
                    {"role": "system", "content": "Answer only yes or no."},
                    {"role": "user", "content": prompt}
                ]

                # Generate the response
                response = client.chat.completions.create(
                    model=model,
                    messages=messages
                )
                answer = response.choices[0].message.content.strip()

                if model == "deepseek-r1:70b":
                    # Remove <think>...</think> completely
                    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

                output_data.append({"rule": rule_text, "model": model, "answer": answer})
            except Exception as e:
                print(f"Error on rule {ix} with model {model}: {e}")
                output_data.append({"rule": rule_text, "model": model, "answer": "Error"})

    # Create DataFrame and save
    output_df = pd.DataFrame(output_data)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv_path, index=False)
    print(f"Output saved to: {output_csv_path}")


rule_path = "/app/filtered_unmatchedrules_with_descriptions.csv"
models = ["llama3.2:latest", "llama3.1:70b", "deepseek-r1:70b", "mistral"]
output_csv_path = "/app/answers/desc_output.csv"

evaluate_rules_with_llms(rule_path, models, output_csv_path)