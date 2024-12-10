# LLM-Based Classification & Inter-Annotator Evaluation
- We experimented with Qwen2.5-Coder:32b, Llama3.2, Mistral, Llama3.1:70b, Llama3.1:8b for the disjointness classification task. To maintain consistent and interpretable outputs, we restricted the length of model responses and filtered out the invalid ones that did not align with "yes" or "no". We employed zero- and few-shot prompting along with Prompt chaining.

- We evaluated the level of agreement among the answers provided by different LLMs when classifying sibling classes as disjoint with Krippendorff's $\alpha$ calculation.

Krippendorff’s Alpha (\(\alpha\)) is calculated as:

\[
\alpha = 1 - \frac{D_o}{D_e}
\]

Where:

- **\(D_o\) - Observed Disagreement**:  
  The ratio of disagreements among the models' ratings.
  \[
  D_o = \frac{\text{Number of disagreements}}{\text{Total number of comparisons}}
  \]

- **\(D_e\) - Expected Disagreement**:  
  The level of disagreement expected by chance based on the distribution of responses, which normalizes the agreement calculation.
  \[
  D_e = 1 - \left( p_1^2 + p_0^2 \right)
  \]

Here:
- \(p_1\) represents the probability of a "yes" response.
- \(p_0\) represents the probability of a "no" response.

## auto_LLMs.py
This code script automates the process of generating responses from a list of Language Learning Models (LLMs) for a given set of prompts stored in a CSV file. It evaluates the models, collects their responses, and saves the results to an output directory.

## process.py
This code generates prompts for all rule head & sibling class pairs particularly for zero- and few-shot prompting methods. Then, it gets results from a series of LLMs per chosen pair.

## chain_process.py
This code generates prompts for all rule head & sibling class pairs particularly for prompt chaining method. Then, it gets results from a series of LLMs per chosen pair.

## prompt_generator.py
This script generates the formatted prompts for Large Language Models (LLMs) by combining rule constants and their candidate sibling classes. The prompts are created based on the templates in LLM prompt types folder, and saved as individual CSV files for each rule constant.

## k_alpha.py
Calculates Krippendorff's alpha per constant and sibling pair answers in the related final tables at /../pos-to-neg/LLM_prompting/sample results by LLMs.
    

