from pathlib import Path
import pandas as pd
from tqdm import tqdm


def prompt_generator(prompt_path: str, data_path: str, output_path: str):
    """
    Generates and writes all LLM prompts for each output label.
    
    Args:
        prompt_path (str): Path to the file containing a generic LLM prompt.
        data_path (str): Path to the file containing labels and their outputs to be substituted.
        output_path (str): Path to the folder where the prompt tables should be written.
    """
    # Read prompt
    prompt_path = Path(prompt_path)
    prompt = prompt_path.read_text()
    
    # Read data
    data_path = Path(data_path)
    data = data_path.read_text()
    lines = data.splitlines()  # Data line by line
    
    output_path = Path(output_path) / "prompts"

    output_path.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists
    
    rule_no = 1  # To enumerate rules
    section_active = False  # Tracks whether we're within a section
    section = []  # List to append lines relevant to the current rule
    constant = None  # Tracks the current constant

    for ix, line in tqdm(enumerate(lines), total=len(lines)):
        if line.startswith("Rule_constant:"):  # Start of a new rule
            if section_active:  # Process the previous section before starting a new one
                process_section(rule_no, constant, section, prompt, output_path)
                rule_no += 1
            # Initialize a new rule
            constant = line.split(" ", 1)[-1].strip().replace(" ", "_")  # Extract constant
            section = []  # Reset section
            section_active = True

        elif line.strip() == "":  # End of a section
            if section_active:  # Only process if a section is active
                process_section(rule_no, constant, section, prompt, output_path)
                rule_no += 1
                section_active = False

        else:  # Add lines to the current section
            section.append(line.strip())
    
    # Process the final section if the file ends without an empty line
    if section_active:
        process_section(rule_no, constant, section, prompt, output_path)


def process_section(rule_no, constant, section, prompt, output_path):
    """
    Processes a section of the input data and writes the generated prompts to a CSV.
    
    Args:
        rule_no (int): The current rule number.
        constant (str): The head constant for the rule.
        section (list): Lines related to the current rule.
        prompt (str): The template for generating prompts.
        output_path (Path): The directory to save the generated CSV.
    
    Returns:
        None
    """
    candidates = []
    prompts = []
    
    if section:
        cand_lst = section[0].split(",")  # Format to get all elements in a list
        for candidate in cand_lst:
            candidates.append(candidate.strip())  # Add each candidate
            prompts.append(
                prompt.replace("<candidate-type-placeholder>", candidate)
                      .replace("<rule-head-constant>", constant)
            )  # Add formatted prompt string

    # Write generated prompts
    df_data = {
        "Rule": rule_no,
        "Rule_Constant": constant,
        "Candidate_Class": candidates,
        "Prompt": prompts,
    }
    pd.DataFrame(df_data).to_csv(output_path / f"{constant}.csv", index=False)

# Example Usage

prompt_path = "/app/LLM_prompting/data/prompt/Zero-shot_prompt.txt"
data_path = "/app/LLM_prompting/data/output.txt"
output_path = "/app/LLM_prompting/data"
prompt_generator(prompt_path=prompt_path, data_path=data_path, output_path=output_path)

