import pandas as pd
import numpy as np

def normalize_response(value):
    """
    Normalize the response to binary:
    - Return 1 if the response contains 'yes' (case-insensitive).
    - Return 0 if the response contains 'no' (case-insensitive).
    - Raise an error if neither 'yes' nor 'no' is found.
    """
    value = str(value).strip().lower()
    if "yes" in value:
        return 1
    elif "no" in value:
        return 0
    else:
        raise ValueError(f"Invalid response: {value}. Expected 'yes' or 'no'.")

def krippendorffs_alpha(csv_path):
    """
    Calculate Krippendorff's alpha per line in a CSV, handling variations in 'Yes'/'No' responses.
    
    Args:
        csv_path (str): Path to the input CSV file.
        
    Returns:
        pd.DataFrame: A DataFrame with Krippendorff's alpha values for each rule.
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Define the output DataFrame to store alpha values
    results = []

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        # Extract the model ratings
        ratings = row.iloc[3:].values  # Ratings start from the 4th column
        
        # Normalize the ratings to binary (1 for 'Yes', 0 for 'No')
        try:
            ratings = [normalize_response(r) for r in ratings]
        except ValueError as e:
            print(f"Skipping row due to invalid response: {e}")
            continue
        
        # Calculate observed disagreement (D_o)
        total_comparisons = len(ratings) * (len(ratings) - 1)
        disagreements = 0
        for i in range(len(ratings)):
            for j in range(i + 1, len(ratings)):
                if ratings[i] != ratings[j]:
                    disagreements += 1
        D_o = disagreements / total_comparisons if total_comparisons > 0 else 0

        # Calculate expected disagreement (D_e)
        p_1 = sum(ratings) / len(ratings)  # Probability of 'Yes'
        p_0 = 1 - p_1  # Probability of 'No'
        D_e = 1 - (p_1**2 + p_0**2)

        # Calculate Krippendorff's alpha
        if D_e == 0:  # Avoid division by zero
            alpha = 1.0 if D_o == 0 else 0.0
        else:
            alpha = 1 - (D_o / D_e)

        # Append results
        results.append({
            "Rule": row["Rule"],
            "Rule_Constant": row["Rule_Constant"],
            "Candidate_Class": row["Candidate_Class"],
            "Krippendorff_Alpha": alpha
        })
    
    # Convert results to a DataFrame and return
    return pd.DataFrame(results)


csv_path = "/app/LLM_prompting/data/answers/few/human_final.csv"  # Replace with the path to your CSV file
result_df = krippendorffs_alpha(csv_path)

# Save the results
result_df.to_csv("/app/LLM_prompting/data/k_alpha/few-shot/krippendorff_alpha_results_human.csv", index=False)

print("Krippendorff's alpha calculation completed and saved to 'krippendorff_alpha_results.csv'.")
