import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
import inngest
import inngest.fast_api
from groq import AsyncGroq
from custom_types import RAGQueryResult, RAGsearchResult, RAGChunkandSRC, RAGUpsertDocumentResult
import uuid
from vector_db import QdrantStorage
from data_loader import embed_chunks, load_and_chunk_pdf, sparse_embed_chunks
import hashlib
from qdrant_client import QdrantClient
from reranker import reranker
from database import save_query_metrics

load_dotenv()


app = FastAPI()

groq_client = AsyncGroq(
    api_key=os.getenv("GROQ_API_KEY")
)


inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
)


@inngest_client.create_function(
    fn_id="rag-ingest-pdf",
    trigger=inngest.TriggerEvent(event="RAG/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context, event: inngest.Event):
    def _load(ctx: inngest.Context, event: inngest.Event) -> RAGChunkandSRC:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkandSRC(chunks=chunks, source_id=source_id)
    
    def _upsert(chunks_and_src: RAGChunkandSRC) -> RAGUpsertDocumentResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id or "unknown"
        vecs = embed_chunks(chunks)
        sparse_vecs = sparse_embed_chunks(
            chunks
        )
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}_{i}")) for i in range(len(chunks))]
        payloads = [{"text": chunks[i], "source_id": source_id} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, sparse_vecs, payloads)
        return RAGUpsertDocumentResult(ingested_chunks=len(chunks))
             
    chunks_and_src = await ctx.step.run("load_pdf", lambda: _load(ctx, event), output_type=RAGChunkandSRC)
    ingested =await ctx.step.run("upsert_chunks", lambda: _upsert(chunks_and_src), output_type=RAGUpsertDocumentResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG Query PDF",
    trigger=inngest.TriggerEvent(event="RAG/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGsearchResult:
        query_vec = embed_chunks([question])[0]
        query_sparse = sparse_embed_chunks([question])[0]
        store = QdrantStorage()
        # found = store.search(query_vec, top_k)
        found = store.search(
            query_vector=query_vec,
            query_sparse=query_sparse,
            top_k=top_k
        )
        return RAGsearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = ctx.event.data.get("top_k", 5)
    found = await ctx.step.run("embed_and_search", lambda: _search(question, top_k), output_type=RAGsearchResult)

    context_results = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = f"""
    Context:
    {context_results}

    Question:
    {question}

    Instructions:
    - Answer only from the provided context.
    - If the answer appears in multiple chunks, combine the information.
    - Quote important legal terms exactly as written.
    - If the answer is not present, reply: "I don't know based on the provided documents."
    """

    res = await ctx.step.ai.infer(
        "llm_answer",
        model="groq:llama-3.1-8b-instant",
        body={"temperature": 0.2, "max_tokens": 1024},
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer only based on the provided context. "
                    "If the answer is not in the context, say you don't know."
                ),
            },
            {"role": "user", "content": user_content},
        ],
    )
    answer = res.choices[0].message.content
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}


inngest.fast_api.serve(
    app,
    inngest_client,
    [rag_ingest_pdf, rag_query_pdf_ai]
)


UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/ingest", response_model=RAGUpsertDocumentResult)
async def ingest_pdf(file: UploadFile = File(...)):
    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    # dest = UPLOAD_DIR / f"{uuid.uuid4()}.pdf"
    dest = UPLOAD_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)

    chunks = load_and_chunk_pdf(str(dest))
    if not chunks:
        raise HTTPException(status_code=400, detail="No text extracted from PDF.")
    
    content = await file.read()
    pdf_hash = hashlib.md5(content).hexdigest()
    vecs = embed_chunks(chunks)
    # source_id = str(dest)
    source_id = pdf_hash

    # print("INGESTING:", file.filename)
    # print("SOURCE ID:", source_id)
    
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}_{i}")) for i in range(len(chunks))]
    payloads = [{"text": chunks[i], "source_id": source_id} for i in range(len(chunks))]
    sparse_vecs = sparse_embed_chunks(chunks)
    QdrantStorage().upsert(ids, vecs,sparse_vecs,payloads)

    # print(f"Inserted {len(ids)} chunks")
    # print(f"Vector dimension: {len(vecs[0])}")
    # print(f"First chunk:\n{chunks[0][:500]}")
    
    return RAGUpsertDocumentResult(ingested_chunks=len(chunks))


@app.post("/query", response_model=RAGQueryResult)
async def query_pdf(question: str, top_k: int = 5, evaluation_type: str = "none"):
    if not question.strip():
        raise HTTPException(status_code=400, detail="question is required.")
    
    import time
    request_start = time.perf_counter()

    query_vec = embed_chunks([question])[0]
    sparse_query = sparse_embed_chunks([question])[0]

    retrieval_start = time.perf_counter() 

    found = QdrantStorage().search(query_vector=query_vec,query_sparse=sparse_query, top_k=top_k)
    
    retrieval_latency = (
    time.perf_counter()
    - retrieval_start
    )
    print(f"Embedding and vector search took {retrieval_latency:.4f} seconds.")

    # contexts = found["contexts"]
    # sources = found["sources"]
    matches = found["matches"]

    pairs = [
        [question, m.text]
        for m in matches
    ]

    rerank_start = time.perf_counter()
    
    scores = reranker.predict(pairs)

    rerank_latency = time.perf_counter() - rerank_start
    print(f"Reranking took {rerank_latency:.4f} seconds.")

    for match, score in zip(matches, scores):
        match.rerank_score = float(score)

    matches.sort(
        key=lambda x: x.rerank_score,
        reverse=True
    )

    matches = matches[:5]

    contexts = [
        m.text
        for m in matches
    ]

    sources = list(
        set(
            m.source_id
            for m in matches
        )
    )
    
    # if not contexts:
    #     return RAGQueryResult(answer="I don't know.", sources=[], num_contexts=0)

    for s in sources:
        print("SOURCE:", s)

    context_results = "\n\n".join(f"- {c}" for c in contexts)
    user_content = (
        "You are an AI assistant for answering questions based on the following context from PDF documents:\n"
        f"{context_results}\n\n"
        "Please provide a concise answer to the following question:\n"
        f"{question}\n"
    )
    llm_start = time.perf_counter()
    res = await groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer only based on the provided context. "
                    "If the answer is not in the context, say you don't know."
                ),
            },
            {"role": "user", "content": user_content},
        ],
    )

    llm_latency = time.perf_counter() - llm_start
    print(f"LLM generation took {llm_latency:.4f} seconds.")

    total_latency = (
        time.perf_counter()
        - request_start
    )

    print(f"Total RAG pipeline latency: {total_latency:.4f} seconds.")
    answer = res.choices[0].message.content or "Hell Yeah!!!."
    
    evaluation_metrics = None
    if evaluation_type == "custom":
        from evaluation import evaluate_rag_response
        evaluation_metrics = await evaluate_rag_response(question, contexts, answer)

    elif evaluation_type == "LLAMA_JUDGE":
        from evaluation import evaluate_with_groq_judge

        evaluation_metrics = await evaluate_with_groq_judge(
            question,
            contexts,
            answer
        )
    
    # print("Evaluation Type:", evaluation_type)
    # print("evaluation_metrics =", evaluation_metrics)
    
    ## Saving the metrics into the sqlite database 
    save_query_metrics(
        question=question,
        answer=answer,

        contexts=contexts,
        sources=sources,

        top_k=top_k,

        retrieval_latency=retrieval_latency,
        rerank_latency=rerank_latency,
        llm_latency=llm_latency,
        total_latency=total_latency,

    evaluation=evaluation_metrics
    )
    return RAGQueryResult(
        answer=answer, 
        sources=sources, 
        num_contexts=len(contexts),
        matches=matches,
        evaluation=evaluation_metrics,
        latency_metrics={
            "retrieval_latency": retrieval_latency,
            "rerank_latency": rerank_latency,
            "llm_latency": llm_latency,
            "total_latency": total_latency
        }
    )   


@app.get("/reset_db")
async def reset_db():

    client = QdrantClient(
        host="localhost",
        port=6333
    )

    client.delete_collection("documents")

    return {"message": "Collection deleted"}

