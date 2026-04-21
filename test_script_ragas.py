"""
RAG API test script using RAGAS + Azure OpenAI.

Prerequisites:
pip install ragas datasets langchain-openai requests python-dotenv

Environment variables:
- AZURE_OPENAI_API_KEY=your_key
- AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
- AZURE_OPENAI_API_VERSION=2024-02-01
- AZURE_OPENAI_DEPLOYMENT_NAME=your-chat-deployment (e.g., gpt-4o)
- AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large (or your embedding deployment)
"""

import json
import os
from typing import Any, Dict, List

import requests
from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from app import retrieve, ingest

load_dotenv()



AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

missing = [
    name
    for name, value in {
        "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_DEPLOYMENT_NAME": AZURE_OPENAI_DEPLOYMENT_NAME,
    }.items()
    if not value
]
if missing:
    raise ValueError(f"Missing required env vars: {', '.join(missing)}")


def normalize_retrieval_context(raw_context: Any) -> List[str]:
    """Convert API context payload into list[str] expected by RAGAS."""
    if raw_context is None:
        return []

    if isinstance(raw_context, str):
        return [raw_context]

    if isinstance(raw_context, dict):
        for key in ("text", "content", "chunk", "document", "page_content"):
            value = raw_context.get(key)
            if value:
                return [str(value)]
        return [json.dumps(raw_context, ensure_ascii=False)]

    if isinstance(raw_context, list):
        normalized: List[str] = []
        for item in raw_context:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                value = None
                for key in ("text", "content", "chunk", "document", "page_content"):
                    if item.get(key):
                        value = item.get(key)
                        break
                normalized.append(
                    str(value) if value is not None else json.dumps(item, ensure_ascii=False)
                )
            else:
                normalized.append(str(item))
        return normalized

    return [str(raw_context)]


golden_dataset: List[Dict[str, Any]] = [
    {
        "test_case_id": "tc001",
        "input": "How does a RAG system pipeline flow from query to answer?",
        "expected_output": (
            "User Query goes to Retriever, then Context is provided to LLM, "
            "and then the LLM generates the Answer."
        ),
    },
    {
        "test_case_id": "tc002",
        "input": "What is tokenisation in Large Language Models?",
        "expected_output": (
            "Tokenisation is the process of breaking text into tokens "
            "(words, subwords, or punctuation) that the model can process."
        ),
    },
    {
        "test_case_id": "tc003",
        "input": "Explain the attention mechanism in LLMs.",
        "expected_output": (
            "The attention mechanism allows a model to relate tokens across "
            "the input, helping capture context and dependencies."
        ),
    },
    {
        "test_case_id": "tc004",
        "input": "What is temperature in LLMs and its effect?",
        "expected_output": (
            "Temperature controls randomness in generation; low values are "
            "more deterministic and high values are more creative."
        ),
    },
    {
        "test_case_id": "tc005",
        "input": "What is the Oracle Problem in AI testing?",
        "expected_output": (
            "The Oracle Problem is the difficulty of defining one strict "
            "correct answer because multiple outputs may be valid."
        ),
    },
]


ingest(["uploads/Breaking_the_Black_Box_Module1.pdf"])

def fetch_rag_answer(query: str) -> Dict[str, Any]:
    """Call your RAG API /retrieve endpoint."""
    response = retrieve(query, top_k=3)
    return response.json()


for test_case in golden_dataset:
    try:
        rag_result = fetch_rag_answer(test_case["input"])

        answer_payload = rag_result.get("answer", {})
        if isinstance(answer_payload, dict):
            actual_output = answer_payload.get("answer") or rag_result.get("answer")
            raw_context = answer_payload.get("context")
        else:
            actual_output = rag_result.get("answer")
            raw_context = rag_result.get("context")

        test_case["actual_output"] = str(actual_output) if actual_output else "No answer returned"
        test_case["retrieval_context"] = normalize_retrieval_context(raw_context)

        print(
            f"Filled {test_case['test_case_id']}: "
            f"{test_case['actual_output'][:100].replace(chr(10), ' ')}..."
        )
    except Exception as exc:
        print(f"Error for {test_case['test_case_id']}: {exc}")
        test_case["actual_output"] = "API Error"
        test_case["retrieval_context"] = []

print("\nGolden dataset with actual outputs:")
print(json.dumps(golden_dataset, indent=2, ensure_ascii=False))

# RAGAS-compatible tabular data
ragas_rows = {
    "question": [item["input"] for item in golden_dataset],
    "answer": [item["actual_output"] for item in golden_dataset],
    "ground_truth": [item["expected_output"] for item in golden_dataset],
    "contexts": [item["retrieval_context"] for item in golden_dataset],
}

dataset = Dataset.from_dict(ragas_rows)

# Azure OpenAI judge model + embeddings used by RAGAS
azure_llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    temperature=0,
)

embedding_deployment = AZURE_OPENAI_EMBEDDING_DEPLOYMENT or AZURE_OPENAI_DEPLOYMENT_NAME
azure_embeddings = AzureOpenAIEmbeddings(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=embedding_deployment,
)

print("\nRunning RAGAS evaluation...")
results = evaluate(
    dataset=dataset,
    metrics=[answer_relevancy, faithfulness, context_recall, context_precision],
    llm=azure_llm,
    embeddings=azure_embeddings,
)

print("\nRAGAS aggregate scores:")
print(results)



print("\nRAGAS per-test-case scores:")
results_df = results.to_pandas()
print(results_df)
results_df.to_csv("ragas_metrics.csv", index=False)

# --- CI assertions ---
# Ensure retrieval/answer generation succeeded for every golden test.
failed_cases = [item["test_case_id"] for item in golden_dataset if item["actual_output"] == "API Error"]
assert not failed_cases, f"RAG retrieval failed for test case(s): {', '.join(failed_cases)}"

empty_context_cases = [
    item["test_case_id"] for item in golden_dataset if not item.get("retrieval_context")
]
assert not empty_context_cases, (
    f"Missing retrieval context for test case(s): {', '.join(empty_context_cases)}"
)

required_metric_columns = {
    "answer_relevancy",
    "faithfulness",
    "context_recall",
    "context_precision",
}
missing_columns = required_metric_columns.difference(results_df.columns)
assert not missing_columns, f"Missing expected RAGAS metric column(s): {sorted(missing_columns)}"

# Default thresholds can be tuned via CI env vars without code changes.
thresholds = {
    "answer_relevancy": float(os.getenv("RAGAS_MIN_ANSWER_RELEVANCY", "0.70")),
    "faithfulness": float(os.getenv("RAGAS_MIN_FAITHFULNESS", "0.70")),
    "context_recall": float(os.getenv("RAGAS_MIN_CONTEXT_RECALL", "0.65")),
    "context_precision": float(os.getenv("RAGAS_MIN_CONTEXT_PRECISION", "0.65")),
}

aggregate_scores = {}
for metric_name in required_metric_columns:
    metric_value = getattr(results, metric_name, None)
    if metric_value is None:
        metric_value = float(results_df[metric_name].mean())
    aggregate_scores[metric_name] = float(metric_value)

for metric_name, threshold in thresholds.items():
    score = aggregate_scores[metric_name]
    assert score >= threshold, (
        f"RAGAS metric '{metric_name}' below threshold: {score:.4f} < {threshold:.4f}"
    )

print("\nRAGAS threshold assertions passed.")
