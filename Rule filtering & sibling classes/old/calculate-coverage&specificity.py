import pandas as pd
import re
from rdflib import Graph, URIRef

def parse_predicate_and_object(predicate_obj_str):
    # Regular expression to match the predicate and object within parentheses
    match = re.match(r"(.+?)\((.+),(.+)\)", predicate_obj_str.strip())
    if match:
        predicate_str = match.group(1).strip()
        obj1_str = match.group(2).strip()
        obj2_str = match.group(3).strip()
        return predicate_str, obj1_str, obj2_str
    else:
        raise ValueError("Unexpected format for predicate and object extraction")


f_df = pd.read_csv(
    "/home/jovyan/work/pos-to-neg-rules/rules-1000-anyb-codex",
    sep='\t',
    header=None,
    names=["score", "frequency", "probability", "rule"]
)

# Certainty Assessment Functions 
g = Graph()
g.parse("/home/jovyan/work/pos-to-neg-rules/codexM-output.ttl", format="turtle")

#Measures how specific the given object (rule_head_obj) is to the predicate, compared to other objects of the predicate.
def calculate_specifity(predicate, rule_head_obj):
    #Counts how many times the rule_head_obj is associated with the given predicate in the KG.
    total_head_instances = len(list(g.triples((None, predicate,rule_head_obj ))))
    #Counts how many times the predicate is used with any object in the KG.
    total_head_var_instances = len(list(g.triples((None, predicate, None))))
    #specificity score of the rule
    return total_head_instances / total_head_var_instances 

# Apply assessments on each rule
def assess_rule(row):
    flag = 0
    rule = row["rule"]
    
    body_predicate_s, body_object1_s, body_object2_s = parse_predicate_and_object(rule.split("<=")[1].strip())
    body_predicate = URIRef(body_predicate_s)
    body_object1 = URIRef(body_object1_s)
    body_object2 = URIRef(body_object2_s)
    
    head_predicate_s, head_object1_s, head_object2_s = parse_predicate_and_object(rule.split("<=")[0].strip())
    head_predicate = URIRef(head_predicate_s)
    head_object1 = URIRef(head_object1_s)
    head_object2 = URIRef(head_object2_s)
    
    uri_regex = r'(http[s]?:\/\/[^\s<>\"\(\),]+)'
    uri_pattern = re.compile(uri_regex)
    
    body_object = body_object1 if uri_pattern.search(body_object1) else (body_object2 if uri_pattern.search(body_object2) else None)
    head_object = head_object1 if uri_pattern.search(head_object1) else (head_object2 if uri_pattern.search(head_object2) else None)
    
    if body_object and head_object:
        #specificity calculation of the rule
        specificity = calculate_specifity(head_predicate,head_object )
        return specificity
    else:
        return 0

# Define confidence ranges and process each range
#first filtering based on confidence score >0.85
ranges = [
    (0.85, 0.90),(0.90, 0.95), (0.95, 1.0)
]

for lower, upper in ranges:
    range_df = f_df[(f_df["probability"] >= lower) & (f_df["probability"] < upper)].copy()
 
    print(f"Range ({lower}, {upper}): {len(range_df)} entries")
    #second filtering
    # Calculate specificity for each rule in the range
    range_df["specificity"] = range_df.apply(assess_rule, axis=1)
    
    output_file_path = f'/home/jovyan/work/pos-to-neg-rules/output_{upper}-{lower}.csv'
    
    # Export the range DataFrame to a CSV file
    range_df.to_csv(output_file_path, sep='\t', index=False)
    print(f"Exported {output_file_path} with {len(range_df)} rows.")
