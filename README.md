# PEARL

![assets/image.png](assets/image.png)

PEARL is an code generation method based on retrieval-augmented generation (RAG), specifically designed for low-resource programming languages.

## Project Overview

While Large Language Models (LLMs) have demonstrated remarkable capabilities in code intelligence, their performance on low-resource programming languages (such as Racket, OCaml, R) remains relatively weak. PEARL aims to address this limitation through retrieval-augmented generation technology.

## Key Features

- **Knowledge Base Construction**: Building specialized knowledge bases for low-resource programming languages.
- **Retriever Distillation**: Employing an distillation method to improve the retrieval model.
- **Retrieval-Augmented Thinking**: We introduce a retrieval-augmented thinking process to enhance the model's ability to generate code based on retrieved context.

## Supported Programming Languages

We support the following programming languages five languages: Racket, OCaml, R, Python, and Java. Now we have release the Racket version, more will be released soon.

## Project Structure

```
PEARL/
├── config/                   # Configuration files
├── methods/                  # different inference methods
│   ├── pearl/                # PEARL Inference
│   ├── raw/                  # raw inference
│   └── ...
├── rag/                      # Retrieval components
│   ├── retriever.py
│   └── embedder.py
├── contrastive_learning/     # RRD training
├── documents/                # Knowledge bases and processing
├── prompts/                  # Prompts
├── utils/
└── multipl_e/                # Evaluation framework
```

## Quick Start
Generate code completions:

```bash
python inference.py lang="rkt" name="unsloth/Meta-Llama-3.1-8B-Instruct" method="pearl"
```

Run evaluation across multiple languages:

```bash
bash run_gen.sh
```