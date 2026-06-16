from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any
from custom_types import RAGChunkMatch
from fastembed import SparseTextEmbedding
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    Distance
)
from qdrant_client import models
from qdrant_client.models import (
    PointStruct,
    SparseVector
)

class QdrantStorage: 
    def __init__(
        self,
        host="localhost",
        port=6333,
        collection_name="documents",
        dim=384,
    ):

        self.client = QdrantClient(
            host=host,
            port=port,
            timeout=30
        )

        self.collection_name = collection_name

        collections = self.client.get_collections().collections

        exists = any(
            c.name == collection_name
            for c in collections
        )

        if not exists:
            self.client.create_collection(
                collection_name=collection_name,

                vectors_config={
                    "dense": VectorParams(
                        size=dim,
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams()
                }
            )

        info = self.client.get_collection(self.collection_name)
        # print("\nCOLLECTION INFO")
        # print(info)
        count = self.client.count(
            collection_name=self.collection_name,
            exact=True
        )
        # print("TOTAL VECTORS:", count.count)
    
    # def upsert(self, ids, vectors, payloads):
    #     points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids) )]
    #     self.client.upsert(collection_name=self.collection_name, points=points)

    def upsert(self,ids,dense_vectors,sparse_vectors,payloads):
        points = []
        for i in range(len(ids)):
            points.append(
                PointStruct(
                    id=ids[i],
                    vector={
                        "dense": dense_vectors[i],
                        "sparse": SparseVector(
                            indices=sparse_vectors[i].indices.tolist(),
                            values=sparse_vectors[i].values.tolist()
                        )
                    },
                    payload=payloads[i]
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_vector, query_sparse, top_k: int = 5):

        search_result = self.client.query_points(
            collection_name=self.collection_name,

            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=50,
                ),

                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_sparse.indices.tolist(),
                        values=query_sparse.values.tolist()
                    ),
                    using="sparse",
                    limit=50
                )
            ],

            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),

            limit=top_k,
            with_payload=True,
        )
        # for p in search_result.points:

        
        contexts = []
        sources = set()
        matches = []

        for r in search_result.points:
            # print("\nSCORE:", r.score)
            # print("PAYLOAD:", r.payload)
            # print("SOURCE:", r.payload.get("source_id"))
            # print("TEXT:", r.payload.get("text","")[:300])


            payload = r.payload or {}
            if not payload:
                continue

            text = payload.get("text", "")
            source_id = payload.get("source_id", "unknown")
            score = float(r.score) if r.score is not None else 0.0

            contexts.append(text)

            if source_id and source_id != "unknown":
                sources.add(source_id)

            matches.append(
                RAGChunkMatch(
                    text=text,
                    source_id=source_id,
                    score=score
                )
            )

        return {
            "contexts": contexts,
            "sources": sorted(sources),
            "matches": matches
        }


