import streamlit as st
import sqlite3
import pandas as pd
import json

PASSWORD = "admin123"

pwd = st.text_input(
    "Admin Password",
    type="password"
)

if pwd != PASSWORD:

    st.stop()


conn = sqlite3.connect(
    "rag_metrics.db"
)

df = pd.read_sql(
    "SELECT * FROM rag_queries",
    conn
)

st.subheader("Recent Queries")

st.dataframe(
    df[
        [
            "created_at",
            "question",
            "hallucination_rate",
            "total_latency"
        ]
    ]
)

selected_id = st.selectbox(
    "Query",
    df["id"]
)

row = df[
    df["id"] == selected_id
].iloc[0]

st.write("Question")
st.code(row["question"])

st.write("Answer")
st.write(row["answer"])


contexts = json.loads(
    row["retrieved_contexts"]
)

for i, context in enumerate(contexts):

    with st.expander(
        f"Chunk {i+1}"
    ):
        st.write(context)

sources = json.loads(
    row["sources"]
)

st.write("Sources")

for source in sources:
    st.write(
        f"- {source}"
    )

st.metric(
    "Total Queries",
    len(df)
)

st.metric(
    "Avg Retrieval Latency",
    f"{df['retrieval_latency'].mean():.3f}s"
)

st.metric(
    "Avg LLM Latency",
    f"{df['llm_latency'].mean():.3f}s"
)

st.metric(
    "Avg LLM Latency",
    f"{df['llm_latency'].mean():.3f}s"
)

import plotly.express as px

fig = px.line(
    df,
    x="created_at",
    y="total_latency",
    title="Total Latency Over Time"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


fig = px.line(
    df,
    x="created_at",
    y="faithfulness",
    title="Faithfulness Trend"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

fig = px.line(
    df,
    x="created_at",
    y="hallucination_rate",
    title="Hallucination Rate"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

## query explorer

st.dataframe(
    df.sort_values(
        "created_at",
        ascending=False
    )
)

