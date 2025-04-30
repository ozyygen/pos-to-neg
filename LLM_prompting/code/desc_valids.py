import pandas as pd

# Load the CSV
input_path = "/app/answers/desc_output.csv" 
df = pd.read_csv(input_path)

# Group by rule
qualified_rules = []

for rule, group in df.groupby("rule"):
    models_with_yes = group[group["answer"].str.lower() == "yes"]["model"].nunique()
    
    # If all 4 models said "yes", keep the rule
    if models_with_yes == 4:
        qualified_rules.append(rule)


result_df = pd.DataFrame(qualified_rules, columns=["rule"])

output_path = "/app/answers/rules_all_models_said_yes.csv"
result_df.to_csv(output_path, index=False)

print(f"Exported {len(result_df)} rules to {output_path}")
