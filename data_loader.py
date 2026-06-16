from huggingface_hub import InferenceClient
from openai import OpenAI
import os
# from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper
# from langchain import OpenAI as LangOpenAI
import logging
from dotenv import load_dotenv
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
import os
from fastembed import SparseTextEmbedding


load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logging.basicConfig(level=logging.INFO)


hf_client = InferenceClient(
    provider="hf-inference",
    api_key=os.getenv("HF_TOKEN")
)
print("Hugging Face Inference Client initialized successfully.")
# print(os.getenv("HF_TOKEN"))
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Example embedding model; replace with your choice
EMBED_DIM = 384

splitter=SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(pdf_path):
    pdf_reader = PDFReader()
    documents = pdf_reader.load_data(file=pdf_path)
    texts= [d.text for d in documents if getattr(d, 'text', None) is not None]
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    embeddings = []
    for chunk in chunks:
        embedding = hf_client.feature_extraction(
            chunk,
            model=EMBEDDING_MODEL
        )
        embeddings.append(embedding)
    return embeddings

sparse_model = SparseTextEmbedding(
    model_name="Qdrant/bm25"
)
def sparse_embed_chunks(chunks):

    return list(
        sparse_model.embed(chunks)
    )


