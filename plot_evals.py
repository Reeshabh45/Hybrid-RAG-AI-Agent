import pandas as pd
import plotly.graph_objects as go

def plot_evaluation_radar(evaluation):

    categories = [
        "Context Relevance",
        "Faithfulness",
        "Answer Relevance",
        "Hallucination Safety"
    ]

    values = [
        float(evaluation.get("context_relevance", 0)),
        float(evaluation.get("faithfulness", 0)),
        float(evaluation.get("answer_relevance", 0)),
        float(evaluation.get("hallucination_rate", 0))
    ]

    eval_type = evaluation.get(
        "evaluation_type",
        "custom"
    )

    # Normalize custom judge scores (1-5) to 0-1
    if eval_type in ["custom", "LLAMA_JUDGE"]:

        max_score = max(values)

        if max_score > 1:
            values = [v / 5.0 for v in values]

    values += values[:1]
    categories += categories[:1]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name="Evaluation"
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        height=500
    )

    return fig