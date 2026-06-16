import asyncio
from pathlib import Path
import time
from unittest import result
import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests
from plot_evals import plot_evaluation_radar

load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)

# client = get_inngest_client()
# print(client.event_api_origin)
# print(client.api_origin)

def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


# async def send_rag_ingest_event(pdf_path: Path) -> None:
#     client = get_inngest_client()
#     await client.send(
#         inngest.Event(
#             name="RAG/ingest_pdf",
#             data={
#                 "pdf_path": str(pdf_path.resolve()),
#                 "source_id": pdf_path.name,
#             },
#         )
#     )

def send_rag_ingest_event(pdf_path: Path):

    with open(pdf_path, "rb") as f:

        response = requests.post(
            "http://localhost:8000/ingest",
            files={
                "file": (
                    pdf_path.name,
                    f,
                    "application/pdf"
                )
            }
        )

    # print(response.status_code)
    # print(response.text)

    response.raise_for_status()

    return response.json()


st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion..."):
        path = save_uploaded_pdf(uploaded)
        # Kick off the event and block until the send completes
        send_rag_ingest_event(path)
        # Small pause for user feedback continuity
        time.sleep(0.3)
    st.success(f"Triggered ingestion for: {path.name}")
    st.caption("You can upload another PDF if you like.")

st.divider()
st.title("Ask a question about your PDFs")


def send_rag_query_event(question: str, top_k: int, evaluation_type: str = "none"):
    # print("Querying...")
    # print("Question:", question)
    # print("Top K:", top_k)
    response = requests.post(
        "http://localhost:8000/query",
        params={
            "question": question,
            "top_k": top_k,
            "evaluation_type": evaluation_type
        }
    )

    # print("STATUS:", response.status_code)
    # print("RESPONSE:", response.text)

    # response.raise_for_status()

    return response.json()

def _inngest_api_base() -> str:
    # Local dev server default; configurable via env
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8000/v1")

def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    # resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])

def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)


with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=50, value=10, step=1)
    eval_mode = st.selectbox("Evaluation Mode", ["none", "custom", "LLAMA_JUDGE"])
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():

        with st.spinner("Generating answer and optionally evaluating..."):

            output = send_rag_query_event(
                question.strip(),
                int(top_k),
                eval_mode
            )

            answer = output.get("answer", "")
            sources = output.get("sources", [])

        st.subheader("Answer")
        st.write(answer or "(No answer)")
        
        evaluation = output.get("evaluation")

        # print("Evaluation:", evaluation)
        # print("Type:", type(evaluation))

        # if evaluation:
        #     st.write("---")
        #     eval_type = evaluation.get("evaluation_type")
        #     st.subheader(f"📊 Evaluation Metrics ({eval_type.upper()})")
        #     col1, col2, col3 = st.columns(3)
            
        #     if eval_type == "LLAMA_JUDGE":
        #         col1.metric("Context Precision", f"{evaluation.get('context_relevance', 0.0):.2f}")
        #         col2.metric("Faithfulness", f"{evaluation.get('faithfulness', 0.0):.2f}")
        #         col3.metric("Answer Relevancy", f"{evaluation.get('answer_relevance', 0.0):.2f}")
        #         overall_score = (
        #             float(evaluation.get("context_relevance", 0)) +
        #             float(evaluation.get("faithfulness", 0)) +
        #             float(evaluation.get("answer_relevance", 0))
        #         ) / 3

        #     else:
        #         col1.metric("Context Relevance", f"{evaluation.get('context_relevance', 0)}/5")
        #         col2.metric("Faithfulness", f"{evaluation.get('faithfulness', 0)}/5")
        #         col3.metric("Answer Relevance", f"{evaluation.get('answer_relevance', 0)}/5")
                
        #     st.info(f"**Reasoning:** {evaluation.get('reasoning', '')}")
        #     matches = output.get("matches", [])

        #     if matches:
        #         st.write("---")
        #         st.subheader("🔍 Retrieval Analysis")

        #         # ==========================
        #         # Sources Section
        #         # ==========================
        #         unique_sources = sorted(
        #             {
        #                 Path(
        #                     m.get("source_id", "unknown")
        #                 ).name
        #                 for m in matches
        #             }
        #         )

        #         st.write("### 📄 Sources")

        #         for src in unique_sources:
        #             st.write(f"- {src}")

        #         st.write("---")

        #         # ==========================
        #         # Retrieved Chunks
        #         # ==========================
        #         st.subheader("📚 Retrieved Chunks")

        #         for idx, match in enumerate(matches):

        #             vector_score = match.get(
        #                 "score",
        #                 0.0
        #             )

        #             rerank_score = match.get(
        #                 "rerank_score",
        #                 None
        #             )

        #             source_id = match.get(
        #                 "source_id",
        #                 "unknown"
        #             )

        #             source_name = (
        #                 Path(source_id).name
        #                 if source_id
        #                 else "unknown"
        #             )

        #             text = match.get(
        #                 "text",
        #                 ""
        #             )

        #             # -------------------------
        #             # Expander Title
        #             # -------------------------
        #             if rerank_score is not None:

        #                 expander_label = (
        #                     f"#{idx+1} | "
        #                     f"Vector={vector_score:.3f} | "
        #                     f"Reranker={rerank_score:.3f}"
        #                 )

        #             else:

        #                 expander_label = (
        #                     f"#{idx+1} | "
        #                     f"Vector={vector_score:.3f}"
        #                 )

        #             # -------------------------
        #             # Expander Content
        #             # -------------------------
        #             with st.expander(expander_label):

        #                 col1, col2 = st.columns(2)

        #                 col1.metric(
        #                     "Vector Similarity",
        #                     f"{vector_score:.3f}"
        #                 )

        #                 if rerank_score is not None:

        #                     col2.metric(
        #                         "Reranker Score",
        #                         f"{rerank_score:.3f}"
        #                     )

        #                 st.caption(
        #                     f"Source: {source_name}"
        #                 )

        #                 st.markdown("#### Chunk Content")

        #                 st.info(text)

        #     elif sources:

        #         st.write("### 📄 Sources")

        #         for s in sources:
        #             st.write(f"- {Path(s).name}")
        
        ## latency metrics display
        print("-----------------------------------------------------")
        # st.write("DEBUG OUTPUT")
        # st.json(output)
        metrics = output.get("latency_metrics", {})
        # print("latency metrics:",metrics)
        if metrics:

            st.write("---")
            st.subheader("⚡ Performance Metrics")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Retrieval",
                f"{metrics.get('retrieval_latency',0):.3f}s"
            )

            c2.metric(
                "Rerank",
                f"{metrics.get('rerank_latency',0):.3f}s"
            )

            c3.metric(
                "LLM",
                f"{metrics.get('llm_latency',0):.3f}s"
            )

            c4.metric(
                "Total",
                f"{metrics.get('total_latency',0):.3f}s"
            )
        
        ##plotly
        try:

            st.write("### 🎯 Evaluation Radar Chart")
            radar_fig = plot_evaluation_radar(
                evaluation
            )

            st.plotly_chart(
                radar_fig,
                use_container_width=True
            )
            st.info(f"**Reasoning:** {evaluation.get('reasoning', '')}")
            
        except Exception as e:
            st.error(f"Radar chart error: {e}")
        



