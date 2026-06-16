import pydantic

class RAGChunkandSRC(pydantic.BaseModel):
    chunks: list[str]
    source_id: str | None = None

class RAGUpsertDocumentResult(pydantic.BaseModel):
    ingested_chunks: int

class RAGChunkMatch(pydantic.BaseModel):
    text: str
    source_id: str
    score: float
    rerank_score: float | None = None

class RAGsearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]
    matches: list[RAGChunkMatch] = []

class EvaluationMetrics(pydantic.BaseModel):
    context_relevance: float
    faithfulness: float
    answer_relevance: float
    reasoning: str = ""
    evaluation_type: str = "custom"
    hallucination_rate: float = 0.0

class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int
    matches: list[RAGChunkMatch] = []
    evaluation: EvaluationMetrics | None = None
    latency_metrics: dict | None = None


# import pkgutil
# import inngest

# print([m.name for m in pkgutil.iter_modules(inngest.__path__)])