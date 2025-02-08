# Hypothesis Generation for Materials Discovery and Design Using Goal-Driven and Constraint-Guided LLM Agents
![figure_1_matsci_increased_font drawio (1) drawio (1)](https://github.com/user-attachments/assets/d2c2c1c7-a5e4-4c79-8696-822a2d5c31e5)
Figure 1: 
Overview of our iterative hypothesis generation and evaluation pipeline. Starting from an input 
prompt and a knowledge graph, the Hypotheses Gener-ator (GPT-4o) proposes 20 hypotheses, which are then
reviewed by three critics–GPT-4o, Claude-3.5-Sonnet, and Gemini-1.5-Flash. Their feedback is consolidated
by the Summarizer (GPT-4o); if unanimous agreement is not reached, the hypotheses along with critic feedback
are fed back to the Hypotheses Generator for refinement and are re-evaluated by the critics. Once approved,
the final hypotheses proceed to the Evaluation Agent (OpenAI-o1-preview) for scoring
