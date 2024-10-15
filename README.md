# Multi-Stage Text Retrieval Pipeline for Question-Answering

## Overview

This project implements a multi-stage text retrieval pipeline for question-answering (Q&A) tasks, leveraging embedding models and ranking models. The pipeline consists of two stages:

1. **Candidate Retrieval**: Embedding models are used to retrieve the top-k relevant passages for a given query.
2. **Reranking**: Ranking models reorder the retrieved passages based on their relevance to the query to improve retrieval accuracy.

The implementation uses the **BEIR benchmark datasets** for evaluation and supports **Natural Questions (NQ)**, **HotpotQA**, and **FiQA** datasets.

## Installation

To install the required dependencies, run:

```bash
pip install transformers sentence-transformers datasets beir
```

## Directory Structure

```bash
.
├── README.md               # This file
├── dataset_preparation.py   # Dataset download and preprocessing
├── retrieval_pipeline.py    # Candidate retrieval with embedding models
├── reranking.py             # Reranking using ranking models
├── evaluation.py            # Benchmarking and NDCG evaluation
└── datasets/                # Folder to store the downloaded datasets
```

## Files

### 1. `dataset_preparation.py`

This script downloads and prepares the datasets from the BEIR benchmark. The datasets are preprocessed into chunks for passage-level retrieval.

**How to run**:

```bash
python dataset_preparation.py
```

### 2. `retrieval_pipeline.py`

Implements candidate retrieval using embedding models. Two embedding models are used: a small model (`all-MiniLM-L6-v2`) and a larger model (`nv-embedqa-e5-v5`).

**How to run**:

```bash
python retrieval_pipeline.py
```

### 3. `reranking.py`

This script reranks the top-k retrieved passages using transformer-based ranking models. Two ranking models are used: `ms-marco-MiniLM-L-12-v2` and `nv-rerankqa-mistral-4b-v3`.

**How to run**:

```bash
python reranking.py
```

### 4. `evaluation.py`

Evaluates the retrieval pipeline using the **NDCG@10** metric, comparing the impact of different embedding and ranking model combinations.

**How to run**:

```bash
python evaluation.py
```

## Example Datasets

The retrieval pipeline uses the following datasets from BEIR for evaluation:

- **Natural Questions (NQ)**
- **HotpotQA**
- **FiQA**

You can specify the dataset you want to use when running the `dataset_preparation.py` script.

## Models Used

### Embedding Models

- **Small Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Large Model**: `nvidia/nv-embedqa-e5-v5`

### Ranking Models

- **Small Ranking Model**: `cross-encoder/ms-marco-MiniLM-L-12-v2`
- **Large Ranking Model**: `nvidia/nv-rerankqa-mistral-4b-v3`

## Results and Analysis

The performance of the retrieval pipeline is measured using **NDCG@10**, which evaluates how well the retrieved and reranked passages align with the relevant ground-truth passages.

After implementing the pipeline, compare the results with and without the ranking models to understand the improvements in retrieval accuracy.
