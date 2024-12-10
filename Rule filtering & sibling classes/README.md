# Rule Filtering & Sibling Class Retrieval
- We filtered the discovered positive rules with confidence and specificity scores.
- We retrieved head constants' sibling classes with SPARQL queries.

## calculate-coverage&specificity.py
The code first filters the rules based on confidence score >0.85. Then calculates specificity per rule.

## sparql-siblingDetermine.txt
The SPARQL query retrieves sibling classes from immadiate parent class of a given entity (referred to as <constant>) within Wikidata.