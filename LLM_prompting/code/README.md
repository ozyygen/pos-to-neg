# LLM-Based Classification
We experimented with Qwen2.5-Coder:32b, Llama3.2, Mistral, Llama3.1:70b, Llama3.1:8b for the disjointness classification task. To maintain consistent and interpretable outputs, we restricted the length of model responses and filtered out the invalid ones that did not align with "yes" or "no". We employed zero- and few-shot prompting along with Prompt chaining.

## auto_LLMs.py
This code script automates the process of generating responses from a list of Language Learning Models (LLMs) for a given set of prompts stored in a CSV file. It evaluates the models, collects their responses, and saves the results to an output directory.

## process.py
This code generates prompts for all rule head & sibling class pairs particularly for zero- and few-shot prompting methods. Then, it gets results from a series of LLMs per chosen pair.

## chain_process.py
This code generates prompts for all rule head & sibling class pairs particularly for prompt chaining method. Then, it gets results from a series of LLMs per chosen pair.

## prompt_generator.py
This script generates the formatted prompts for Large Language Models (LLMs) by combining rule constants and their candidate sibling classes. The prompts are created based on the templates in LLM prompt types folder, and saved as individual CSV files for each rule constant.


