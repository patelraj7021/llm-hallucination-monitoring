# Monitoring LLM correctness 

## Abstract

An increase in LLM capabilities through scaling and the introduction of reasoning chains has led to an increase in model correctness since the early iterations of GPT models, but the ever-present possibility of incorrect outputs remains a key hazard as LLM-based AIs are integrated into workflows.
As opposed to increasing correctness through enhanced capabilities, an alternative approach is to monitor LLM outputs for incorrectness.
Upon detection, an incorrect token could be flagged to the user or the model could be prompted to re-generate the affected tokens.
A decreased rate of incorrect outputs leads to safer and more reliable models, enabling greater confidence when deploying upon them for critical tasks.

This project takes an empirical approach to correctness monitoring in the specific context of simple math problems.
Analyzing the hidden embeddings of a transformer model after each layer reveals trends in the path taken by an embedding as it transforms through the model;
embeddings corresponding to tokens that are factual in nature take a path distinct than embeddings corresponding to incorrect tokens.
The presence of this divergence enables classifier models to predict whether a token will be correct/incorrect before it is presented to the user.

Of the classifiers tested (logistic regression, SVM, random forest, gradient boosted trees), the gradient boosted tree method performed the best.
Accuracy is not a meaningful metric to gauge usefulness here due to the severe class imbalance between correct/incorrect labels.
A gradient boosting classifier achieves 73% recall for incorrect tokens, indicating that is feasible to flag most incorrect tokens in practice.
Precision is limited to 8% however, which degrades the trustworthiness of incorrect flags.

Next steps for this work include: expanding scope from math problems to natural language, generalizing beyond standardized message templates, considering larger (more capable) models, shifting focus from correctness to deception.

## Repo Structure

The `project_report.md` file contains more details on this project.
Due to the computational requirements for this work, I used Google's Colab service.
Code is available in the `colab_notebook.py` file, and it can be opened directly on Colab using its link and the Colab "Open notebook from GitHub" feature.