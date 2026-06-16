# 🚀 Hybrid RAG AI Agent

A production-style Retrieval Augmented Generation (RAG) system built with modern AI infrastructure.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-orange)
![Groq](https://img.shields.io/badge/Groq-LLM-purple)

---

## 📌 Features

### Document Ingestion

- Upload PDF documents
- Automatic text extraction
- Intelligent chunking
- Dense embedding generation
- Sparse BM25 embedding generation
- Qdrant indexing

---

### Hybrid Retrieval

Combines:

✅ Dense Semantic Search (BGE Small)

✅ Sparse Keyword Search (BM25)

✅ Reciprocal Rank Fusion (RRF)

Result:

- Better recall
- Better keyword matching
- Better semantic understanding

---

### Cross Encoder Reranking

Uses:

```python
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Reranks retrieved chunks before sending them to the LLM.

Benefits:

- Higher answer quality
- Reduced hallucinations
- Better context selection

---

### LLM Generation

Powered by:

- Groq
- Llama 3.1 8B Instant

Features:

- Context grounded responses
- Source attribution
- Low latency inference

---

### Evaluation Layer

Supports:

- Custom Evaluation
- Llama-as-a-Judge Evaluation

Metrics:

- Context Relevance
- Faithfulness
- Answer Relevance
- Hallucination Rate

---

### Analytics

Tracks:

- Retrieval Latency
- Rerank Latency
- LLM Latency
- Total Response Time

---

## 🏗️ System Architecture

```text
                ┌─────────────┐
                │ PDF Upload  │
                └──────┬──────┘
                       │
                       ▼
               ┌──────────────┐
               │ Chunking     │
               └──────┬───────┘
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼

 Dense Embeddings           Sparse Embeddings
 (BGE Small)                     (BM25)

        ▼                           ▼
        └─────────────┬─────────────┘
                      ▼

                Hybrid Search
                     (Qdrant)

                      ▼

              Cross Encoder
                 Reranker

                      ▼

                 Top Contexts

                      ▼

              Groq Llama 3.1

                      ▼

                 Final Answer

                      ▼

                 Evaluation

                      ▼

             Analytics Storage
```

---

## 🛠️ Tech Stack

| Component | Technology |
|------------|------------|
| Backend | FastAPI |
| Frontend | Streamlit |
| Vector Database | Qdrant |
| Dense Embeddings | BAAI/bge-small-en-v1.5 |
| Sparse Retrieval | BM25 |
| Reranker | Cross Encoder MiniLM |
| LLM | Groq Llama 3.1 |
| Evaluation | LLM Judge |
| Analytics | SQLite |

---

## 📂 Project Structure

```text
Hybrid-RAG-AI-Agent
│
├── main.py
├── streamlit_app.py
├── vector_db.py
├── data_loader.py
├── reranker.py
├── evaluation.py
├── database.py
├── custom_types.py
│
├── pages
│   └── admin_dashboard.py
│
├── uploads
├── models
├── qdrant_storage
│
├── requirements.txt
├── README.md
└── .env.example
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/Reeshabh45/Hybrid-RAG-AI-Agent.git

cd Hybrid-RAG-AI-Agent
```

### Create Environment

```bash
python -m venv rag_env

rag_env\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create:

```env
.env
```

Add:

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

---

## ▶️ Run Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

---

## ▶️ Run Backend

```bash
uvicorn main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

---

## ▶️ Run Frontend

```bash
streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

---

## 📊 Evaluation Dashboard

Provides:

- Retrieval Analysis
- Similarity Scores
- Reranker Scores
- Radar Chart Metrics
- Hallucination Tracking
- Latency Monitoring

---

## 🎯 Future Improvements

- [ ] Multi PDF Collections
- [ ] User Authentication
- [ ] Langfuse Observability
- [ ] Redis Caching
- [ ] PostgreSQL Analytics
- [ ] Kubernetes Deployment
- [ ] CI/CD Pipeline


## 👨‍💻 Author

**Reeshabh Kumar**

LinkedIn:
(Add LinkedIn URL)

GitHub:

:contentReference[oaicite:0]{index=0}

---

## ⭐ If you found this useful

Give the repository a star ⭐