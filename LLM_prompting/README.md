# LLM-Based Inter-Annotator Evaluation
- We experimented with Llama3.1:70b, TBD for the disjointness classification task. To maintain consistent and interpretable outputs, we restricted the length of model responses and filtered out the invalid ones that did not align with "yes" or "no". We employed zero- and few-shot prompting along with Prompt chaining.

* We provide a sample sibling_output.txt which containts target relations along with their candidate disjoint properties. Note rule_constant is acually is rule head relation, and chosen_rules is the target head relation. We'll update them in the codes soon.

## auto_LLMs.py
This code script automates the process of generating responses from a list of Language Learning Models (LLMs) for a given set of prompts stored in a CSV file. It evaluates the models, collects their responses, and saves the results to an output directory.

## process.py
This code generates prompts for all rule head & sibling class pairs particularly for zero- and few-shot prompting methods. Then, it gets results from a series of LLMs per chosen pair.



