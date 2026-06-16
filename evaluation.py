import os
import json
from groq import AsyncGroq
from custom_types import EvaluationMetrics
from dotenv import load_dotenv
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric
)
from deepeval.test_case import LLMTestCase
import json

load_dotenv()

groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

EVALUATION_PROMPT = """
                You are an expert evaluator grading a Retrieval-Augmented Generation (RAG) system.
                You will be provided with a Question, Retrieved Contexts, and a Generated Answer.

                Please evaluate the following 3 metrics on a scale of 1 to 5:
                1. Context Relevance: Does the retrieved context contain the information needed to answer the question? (1 = completely irrelevant, 5 = contains exact answer).
                2. Faithfulness: Is the generated answer grounded ONLY in the retrieved context without making up facts? (1 = hallucinated/not grounded, 5 = perfectly faithful to context).
                3. Answer Relevance: Does the generated answer directly and accurately address the user's question? (1 = completely off-topic, 5 = perfectly answers the question).
                4. Hallucination Rate: What percentage of the answer is hallucinated? (0 = no hallucination, 100 = fully hallucinated).
                You must respond with ONLY a valid JSON object matching this schema:
                {{
                    "context_relevance": <int>,
                    "faithfulness": <int>,
                    "answer_relevance": <int>,
                    "hallucination_rate": <int>,
                    "reasoning": "<short string explaining the scores>"
                }}

                Question: {question}

                Retrieved Contexts:
                {contexts}

                Generated Answer:
                {answer}
                """

async def evaluate_rag_response(question: str, contexts: list[str], answer: str) -> EvaluationMetrics:
    context_str = "\n\n".join([f"Context {i+1}:\n{c}" for i, c in enumerate(contexts)])
    prompt = EVALUATION_PROMPT.format(
        question=question,
        contexts=context_str,
        answer=answer
    )
    
    try:
        res = await groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        result_text = res.choices[0].message.content
        data = json.loads(result_text)
        return EvaluationMetrics(
            context_relevance=float(data.get("context_relevance", 0)),
            faithfulness=float(data.get("faithfulness", 0)),
            answer_relevance=float(data.get("answer_relevance", 0)),
            hallucination_rate=float(data.get("hallucination_rate", 0)),
            reasoning=str(data.get("reasoning", "")),
            evaluation_type="custom"
        )
    except Exception as e:
        print(f"Evaluation error: {e}")
        return EvaluationMetrics(
            context_relevance=0.0,
            faithfulness=0.0,
            answer_relevance=0.0,
            hallucination_rate=0.0,
            reasoning=f"Error during evaluation: {e}",
            evaluation_type="custom"
        )

# DeepEval
# async def evaluate_with_ragas(question: str, contexts: list[str], answer: str) -> EvaluationMetrics:
#     from ragas import evaluate
#     from ragas.metrics import faithfulness, answer_relevancy, context_precision
#     from datasets import Dataset
#     from langchain_groq import ChatGroq
#     from langchain_huggingface import HuggingFaceEmbeddings
    
#     try:
#         groq_llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
#         hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
#         data = {
#             "question": [question],
#             "contexts": [contexts],
#             "answer": [answer],
#             "ground_truth": [""] # Dummy ground truth as it's required by some metrics structurally but not used by faithfulness/relevancy
#         }
#         dataset = Dataset.from_dict(data)
        
#         result = evaluate(
#             dataset=dataset,
#             metrics=[faithfulness, answer_relevancy, context_precision],
#             llm=groq_llm,
#             embeddings=hf_embeddings,
#         )
        
#         # Result is typically a dictionary containing the scores
#         return EvaluationMetrics(
#             context_relevance=float(result.get("context_precision", 0.0)),
#             faithfulness=float(result.get("faithfulness", 0.0)),
#             answer_relevance=float(result.get("answer_relevancy", 0.0)),
#             reasoning="RAGAS automated scoring (scores are normalized 0-1).",
#             evaluation_type="ragas"
#         )
#     except Exception as e:
#         print(f"RAGAS Evaluation error: {e}")
#         return EvaluationMetrics(
#             context_relevance=0.0,
#             faithfulness=0.0,
#             answer_relevance=0.0,
#             reasoning=f"Error during RAGAS evaluation: {e}",
#             evaluation_type="ragas"
#         )

# DeepEval
# async def evaluate_with_deepeval(
#     question: str,
#     contexts: list[str],
#     answer: str
# ):

#     test_case = LLMTestCase(
#         input=question,
#         actual_output=answer,
#         retrieval_context=contexts
#     )

#     faithfulness = FaithfulnessMetric(
#         threshold=0.7
#     )

#     answer_relevancy = AnswerRelevancyMetric(
#         threshold=0.7
#     )

#     context_precision = ContextualPrecisionMetric(
#         threshold=0.7
#     )

#     faithfulness.measure(test_case)
#     answer_relevancy.measure(test_case)
#     context_precision.measure(test_case)

#     return {
#         "context_relevance": context_precision.score,
#         "faithfulness": faithfulness.score,
#         "answer_relevance": answer_relevancy.score,
#         "reasoning": (
#             f"Faithfulness: {faithfulness.reason}\n\n"
#             f"Answer Relevancy: {answer_relevancy.reason}\n\n"
#             f"Context Precision: {context_precision.reason}"
#         ),
#         "evaluation_type": "deepeval"
#     }

# LLM-as-a-Judge evaluator with Groq
async def evaluate_with_groq_judge(
    question: str,
    contexts: list[str],
    answer: str
):

    context_text = "\n\n".join(contexts[:5])

    prompt = f"""
            You are an expert RAG evaluator.

            Question:
            {question}

            Retrieved Context:
            {context_text}

            Generated Answer:
            {answer}

            Evaluate:

            1. Context Relevance (1-5)
            2. Faithfulness (1-5)
            3. Answer Relevance (1-5)

            4. Hallucination Rate (0-100)

            Hallucination Rate Rules:

            0 =
            Every claim is fully supported by context.

            25 =
            Minor unsupported details.

            50 =
            Some unsupported statements.

            75 =
            Many unsupported statements.

            100 =
            Answer mostly invented.

            Return ONLY JSON:

            {{
                "context_relevance": 0,
                "faithfulness": 0,
                "answer_relevance": 0,
                "hallucination_rate": 0,
                "reasoning": "<short string explaining the scores>"
            }}
        """

    response = await groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response.choices[0].message.content

    try:
        result = json.loads(content)

        return EvaluationMetrics(
            context_relevance=float(result.get("context_relevance", 0)),
            faithfulness=float(result.get("faithfulness", 0)),
            answer_relevance=float(result.get("answer_relevance", 0)),
            hallucination_rate=float(result.get("hallucination_rate", 0)),
            reasoning=str(result.get("reasoning", "")),
            evaluation_type="LLAMA_JUDGE"
        )
    except Exception as e:

        return EvaluationMetrics(
            context_relevance=float(result.get("context_relevance", 0)),
            faithfulness=float(result.get("faithfulness", 0)),
            answer_relevance=float(result.get("answer_relevance", 0)),
            hallucination_rate=float(result.get("hallucination_rate", 0)),
            reasoning=str(result.get("reasoning", "")),
            evaluation_type="LLAMA_JUDGE"
        )
# {
#             "context_relevance": result.get("context_relevance", 0),
#             "faithfulness": result.get("faithfulness", 0),
#             "answer_relevance": result.get("answer_relevance", 0),
#             "hallucination_rate": result.get("hallucination_rate", 0),
#             "reasoning": result.get("reasoning", ""),
#             "evaluation_type": "LLAMA_JUDGE"
#         }