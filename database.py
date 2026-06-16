import sqlite3

def init_db():

    conn = sqlite3.connect("rag_metrics.db")

    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS rag_queries (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    question TEXT,

    answer TEXT,

    retrieved_contexts TEXT,

    sources TEXT,

    top_k INTEGER,

    retrieval_latency REAL,

    rerank_latency REAL,

    llm_latency REAL,

    total_latency REAL,

    context_relevance REAL,

    faithfulness REAL,

    answer_relevance REAL,

    hallucination_rate REAL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


import sqlite3

def save_query_metrics(
    question,
    answer,

    contexts,
    sources,

    top_k,

    retrieval_latency,
    rerank_latency,
    llm_latency,
    total_latency,

    evaluation
):

    conn = sqlite3.connect(
        "rag_metrics.db"
    )

    cur = conn.cursor()

    cur.execute(
    """
    INSERT INTO rag_queries (

        question,
        answer,

        retrieved_contexts,
        sources,

        top_k,

        retrieval_latency,
        rerank_latency,
        llm_latency,
        total_latency,

        context_relevance,
        faithfulness,
        answer_relevance,
        hallucination_rate

    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    (
        question,
        answer,

        contexts_json,
        sources_json,

        top_k,

        retrieval_latency,
        rerank_latency,
        llm_latency,
        total_latency,

        evaluation.context_relevance
            if evaluation else None,

        evaluation.faithfulness
            if evaluation else None,

        evaluation.answer_relevance
            if evaluation else None,

        evaluation.hallucination_rate
            if evaluation else None
    )
)
    conn.commit()
    conn.close()