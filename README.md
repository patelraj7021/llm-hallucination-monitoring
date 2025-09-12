# Monitoring LLM Hallucinations

An increase in LLM capabilities through scaling and the introduction of reasoning chains has led to fewer hallucinations since the early iterations of GPT models, but they still remain a key hazard as LLM-based AIs are integrated into workflows.
As opposed through decreasing their frequency through enhanced capabilties and fine tuning, an alternative approach is to monitor LLM outputs for hallucination.
Upon detection, the hallucination could be flagged to the user or the model could be prompted to re-generate the affected tokens.
A decreased rate of hallucations leads to safer and more reliable models, enabling greater confidence when deploying upon them for critical tasks.

This project takes a bottom-up approach to hallucinatin monitoring.
By analyzing the hidden embeddings of a transformer model, a monitoring routine can be utilized to detect trends in the path taken by an embedding as it transforms through the model.
Embeddings corresponding to tokens that are factual in nature take a path distinct than embeddings corresponding to hallucinatory tokens, and this divergence can be used to classify tokens which are more likely to be hallucinations.