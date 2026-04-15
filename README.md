# Commonsense Question Generation using Large Language Models

An end-to-end pipeline leveraging Large Language Models (LLMs) to generate context-aware, inference-driven questions. This repository contains the implementation for an MSc Applied Statistics and Data Science project, integrating explicit graph reasoning with structural skeleton learning to control question difficulty and depth.

## 🚀 Key Features

* **Graph Reasoning Module:** Utilises AMR (Abstract Meaning Representation) parsing and entity extraction to retrieve relational commonsense knowledge from the ConceptNet API.
* **Structure Learning Module:** Employs a Hidden Semi-Markov Model (HSMM) to mine question skeletons, enabling controllable generation based on specific reasoning hop counts (difficulty levels).
* **Multi-Model Integration:** Supports both supervised fine-tuning and zero-shot generation across multiple LLM architectures.
* **Local Caching Mechanism:** Efficient JSON-based caching for ConceptNet API queries to speed up repeated processing without heavy NLP dependencies.

## 🧠 Models & Datasets

**Supported Model Backends:**
* **Fine-Tuned:** `BART` (Sequence-to-Sequence)
* **Zero-Shot LLMs:** `Mistral-7B-Instruct-v0.3`, `Qwen-2.5-7B-Instruct`, `Gemini Flash Lite 2.5`

**Evaluation Datasets:**
* **CosmosQA:** Reading comprehension with commonsense reasoning.
* **MCScript:** Narrative texts concerning everyday activities.
* **KQAPro:** Complex reasoning over knowledge bases.
* **GrailQA:** Strongly generalisable question answering.

## 📊 Evaluation Metrics
Generated outputs are evaluated on both lexical overlap and semantic similarity using:
* BLEU-4
* ROUGE-L
* METEOR
* BERTScore

## ⚙️ Installation & Setup

1. **Clone the repository:**
```bash 
git clone https://github.com/mscasa223/commonsense_qg.git
cd commonsense_qg
