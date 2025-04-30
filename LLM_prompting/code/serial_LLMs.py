from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re
from openai import OpenAI
import os

def cpt_to_desc(code: str, desc: Path) -> str:
    """
    Converts a CPT procedure code to its description.
    
    Args:
        A CPT procedure code from MIMIC-III (code: str) ,
        Path to the CPT code descriptions from MIMIC-III named D_CPT.csv (desc: pathlib Path object) .
        
    Returns:
        The CPT code's description (code_desc: str) .
    """
    
    desc = pd.read_csv(desc) # read description table
    code_desc = "No description available" # default case where code is not defined in the table
    
    for _, row in desc.iterrows():
        desc_range = range(int(row["MINCODEINSUBSECTION"]), int(row["MAXCODEINSUBSECTION"])) # get section range per row
        
        if int(code) in desc_range: # determine if the input falls in the range
            code_desc = row["SUBSECTIONHEADER"]
            break
            
    return code_desc

def icd9_to_desc(code: str, desc: Path) -> str:
    """
    Converts an ICD-9 diagnosis code to its description.

    """
    
    desc = pd.read_csv(desc) # read description table
    code_desc = "No description available" # default case where code is not defined in the table
    
    for _, row in desc.iterrows():
        desc_code = str(row["ICD9_CODE"]) # get row's code
        
        if str(code) == desc_code: # determine if the codes match
            code_desc = row["LONG_TITLE"]
            break
            
    return code_desc

def prompt_generator(prompt_path: str,
                     data_path: str,
                     diagnosis_descriptions_path: str,
                     procedure_descriptions_path: str,
                     output_path: str):
    """
    Generates and writes all LLM prompts for each output label.

    """
    
    # Read prompt
    prompt_path = Path(prompt_path)
    prompt = prompt_path.read_text()
    
    # Read data
    data_path = Path(data_path)
    data = pd.read_csv(data_path, sep=" <= ", header=None, engine="python")
    
    d_icd = Path(diagnosis_descriptions_path)
    d_cpt = Path(procedure_descriptions_path)
    
    output_path = Path(output_path)

    for ix, row in tqdm(data.iterrows()):
        head, body = row.iloc[0], row.iloc[1]
        head = head.replace("https://biomedit.ch/rdf/sphn-ontology/sphn#", "SPHN:")
        head = head.replace("https://biomedit.ch/rdf/sphn-ontology/AIDAVA/", "aidava-resource:")
        
        body = body.replace("https://biomedit.ch/rdf/sphn-ontology/AIDAVA/", "aidava-resource:")
        body = body.replace("https://biomedit.ch/rdf/sphn-resource/icd-9-gm/2023/3/", "ICD9:")
        body = body.replace("https://www.aapc.com/codes/cpt-codes/", "CPT:")
        
        # Filter out Outliers
        age_group = head.split("AgeGroup/")[-1][:-1]
        if age_group == "Outlier":
            continue
        
        code_desc = "No matching description"
        if "ICD9:" in body:
            code = re.search(r"ICD9:([^,]+),X\)", body).group(1).strip()
            code_desc = icd9_to_desc(code, d_icd)
            code = f"ICD9:{code}"
        elif "CPT:" in body:
            code = re.search(r"CPT:([^,]+),X\)", body).group(1).strip()
            code_desc = cpt_to_desc(code, d_cpt)
            code = f"CPT:{code}"
        else:
            code = ":"
        
        new_prompt = prompt.format(body=body,
                                   head=head,
                                   code=code,
                                   code_desc=code_desc)
        
        with open(output_path / f"horn_rule_{ix}.txt", "w") as f:
            f.write(new_prompt)
            
def get_LLM_answer(question: str, model: str) -> str:
    """
    Given a question, get an answer from the specified LLM model.

    """
    
    # Define LLM client
    client = OpenAI(
    base_url="path",
    api_key="key"
    )
    
  
    messages = [
        {"role": "system", "content": f"Answer only in 'Yes', 'No', or 'NA'."},
        {"role": "user", "content": question}
    ]
    
    response = client.chat.completions.create(
                model=model,
                messages=messages,
                
    )
    answer = response.choices[0].message.content # get answer
    
    return answer

def auto_LLMs(prompts_path: str, models: list, output_path: str):
    """
    For a given list of LLMs and several prompts in a given folder, answers are generated, recorded, and written to a CSV file.
 
    """

    # Get rule prompts
    prompts_path = Path(prompts_path)
    prompts = [prompt for prompt in os.listdir(prompts_path) if prompt.endswith(".txt")]
    
    output_path = Path(output_path) # get output folder as path
    
    questions = []
    for prompt in prompts:
        question = (prompts_path / prompt).read_text() # get each prompt text
        questions.append(question)
        

    results = {} 
    for model in models:
        print(f"Prompting {model} for answers...")
        answers = []
        
        for question in tqdm(questions):
            answer = get_LLM_answer(question=question, model=model) # get answer from this model
            
            if model == "deepseek-r1:70b":
                
                answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
                
                answers.append(answer)          
            else:
                answers.append(answer)
            
        results[model] = answers
    
    res_df = pd.DataFrame(results)
       
    res_df.to_csv(output_path / "output.csv",index=False)
    
def serial_LLMs(template_path: str,
                data_path: str,
                diagnosis_descriptions_path: str,
                procedure_descriptions_path: str,
                prompts_path: str,
                models: list,
                output_path: str):
    """
    Given a prompt template, horn rule table, and list of LLMs; format prompts, generate answers, and output as a table.

    """

    print("Generating Prompts...")
    prompt_generator(prompt_path=template_path,
                     data_path=data_path,
                     
                   diagnosis_descriptions_path=diagnosis_descriptions_path,
                    procedure_descriptions_path = procedure_descriptions_path,
                    output_path=prompts_path)
    
    print("Getting answers...")
    auto_LLMs(prompts_path=prompts_path, models=models, output_path=output_path)
    
    print("Done.")
    

template_path = "/app/prompt.txt"
data_path = "/app/horn_rules.csv"
diagnosis_descriptions_path = "/app/D_ICD_DIAGNOSES.csv"
procedure_descriptions_path = "/app/D_CPT.csv"
prompts_path = "/app/prompts"

models = [ "llama3.2:latest","llama3.1:70b","deepseek-r1:70b","mistral"]
output_path = "/app/answers"

serial_LLMs(template_path=template_path,
            data_path=data_path,
            diagnosis_descriptions_path=diagnosis_descriptions_path,
            procedure_descriptions_path=procedure_descriptions_path,
            prompts_path=prompts_path,
            models=models,
            output_path=output_path)