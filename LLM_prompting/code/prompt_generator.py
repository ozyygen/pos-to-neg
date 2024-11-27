from pathlib import Path
import pandas as pd
from tqdm import tqdm

def prompt_generator(prompt_path: str, data_path: str, output_path: str):
    """
    Generates and writes all LLM prompts for each output label.
    
    Args:
        Path to the file containing a generic LLM prompt (prompt_path: str) ,
        Path to the file containing labels and their outputs to be substitutes (data_path: str) ,
        Path to the folder where the prompt tables should be written (output_path: str) .
    
    Returns:
        None .
    """
    
    # Read prompt
    prompt_path = Path(prompt_path)
    prompt = prompt_path.read_text()
    
    # Read data
    data_path = Path(data_path)
    data = data_path.read_text()
    lines = data.splitlines() # data line by line
    
    output_path = Path(output_path)
 
    rule_no = 1 # to enumerate rules
    for ix, line in tqdm(enumerate(lines)):
        if line.startswith("Rule_constant:"): # find rule constant
            
            # Lists for the DataFrame
            candidates = []
            prompts = []
  
            constant = line.split(" ", 1)[-1].strip().replace(" ", "_") # get constant name
            section = [] # list to append lines relevant to the constant
            
            for nline in lines[ix+1:]:
                
                # If line is not the whitespace before the next rule
                if nline != "":
                    section.append(nline) # add relevant line to the section list
                
                # If the relevant section has ended
                else:
                    cand_lst = section[0].split(",") # format to get all elements in a list
                    for candidate in cand_lst:
                        candidates.append(candidate.strip()) # add each candidate
                        prompts.append(prompt.replace("<candidate-type-placeholder>", candidate).replace("<rule-head-constant>", constant)) # add formatted prompt string
                    
                    # Write generated prompts
                    df_data = {"Rule": rule_no, "Rule_Constant": constant, "Candidate_Class": candidates, "Prompt": prompts}
                    pd.DataFrame(df_data).to_csv(output_path / f"{constant}.csv", index=None) # write prompts to output path
                    
                    rule_no+=1 # next rule
                    break

# Example

prompt_path = "/home/jovyan/work/pos-to-neg-rules/LLM/DATA/Zero-shot_prompt.txt"
data_path = "/home/jovyan/work/pos-to-neg-rules/LLM/DATA/sibling_output.txt"
output_path = "/home/jovyan/work/pos-to-neg-rules/LLM/prompts"
prompt_generator(prompt_path=prompt_path, data_path=data_path, output_path=output_path)
