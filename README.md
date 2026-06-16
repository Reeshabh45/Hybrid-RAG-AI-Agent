# Hybrid RAG AI Agent

Production-style Retrieval Augmented Generation (RAG) system built using:

- FastAPI
- Streamlit
- Qdrant
- Hybrid Search (Dense + Sparse)
- Cross Encoder Reranking
- Groq LLM
- Custom RAG Evaluation
- SQLite Analytics Dashboard

## Architecture

PDF Upload
↓
Chunking
↓
Dense Embeddings (BGE)
+
Sparse Embeddings (BM25)
↓
Qdrant Hybrid Search
↓
Cross Encoder Reranker
↓
Groq LLM
↓
Evaluation Layer
↓
Analytics Dashboard

## Features

- PDF ingestion
- Hybrid retrieval
- Cross encoder reranking
- Retrieval inspection
- RAG evaluation
- Latency tracking
- Admin analytics dashboard

## Installation

```bash
git clone <repo_url>

cd RAG_AI_Agent

pip install -r requirements.txt
```

## Run FastAPI

```bash
uvicorn main:app --reload
```

## Run Streamlit

```bash
streamlit run streamlit_app.py
```

## Run Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```