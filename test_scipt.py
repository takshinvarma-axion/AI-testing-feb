# Complete Python script to test RAG API with DeepEval using Azure OpenAI
# Prerequisites:
# pip install deepeval requests openai python-dotenv
# Set environment variables:
# RAG_BASE_URL=https://your-rag-api.com
# AZURE_OPENAI_API_KEY=your_key
# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
# AZURE_OPENAI_API_VERSION=2024-02-01
# AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment (e.g., gpt-4o)

import os
import requests
import json
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRecallMetric,ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import AzureOpenAIModel, OllamaModel

load_dotenv()

# RAG API base URL (set your endpoint)
RAG_BASE_URL = os.getenv("RAG_BASE_URL")
if not RAG_BASE_URL:
    raise ValueError("Set RAG_BASE_URL in .env")

# Azure OpenAI client for DeepEval (DeepEval uses this for evaluation)

model_provider = os.geten("model_provider")

azure_client = AzureOpenAIModel()

ollama_client = OllamaModel(model='llama3',base_url='27.0.0.1:11434')

if model_provider=='ollama':
    client = ollama_client
elif model_provider=='azure':
    client = azure_client



def normalize_retrieval_context(raw_context):
    """
    Convert API context payload into list[str] expected by DeepEval.
    Supports strings, dict chunks, and mixed lists.
    """
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
        normalized = []
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

# Golden dataset structure - expanded from paste.txt content with relevant scenarios for AI/RAG testing doc
golden_dataset = [
    {
        "test_case_id": "tc001",
        "input": "How does a RAG system pipeline flow from query to answer?",
        "expected_output": "User Query goes to Retriever, then Context is provided to LLM, and then the LLM generates the Answer.",
        "type": "factoid",
        "difficulty": "easy",
        "tags": ["pipeline", "rag"]
    },
    {
        "test_case_id": "tc002",
        "input": "What is tokenisation in Large Language Models?",
        "expected_output": "Tokenisation is the process of breaking text into tokens (words, subwords, or punctuation) that the model can process mathematically.",
        "type": "definition",
        "difficulty": "easy",
        "tags": ["llm", "tokenisation"]
    },
    {
        "test_case_id": "tc003",
        "input": "Explain the attention mechanism in LLMs.",
        "expected_output": "The attention mechanism allows the model to relate any token to any other token in the input, capturing context and dependencies like resolving pronouns.",
        "type": "explanation",
        "difficulty": "medium",
        "tags": ["attention", "llm"]
    },
    {
        "test_case_id": "tc004",
        "input": "What is temperature in LLMs and its effect?",
        "expected_output": "Temperature is a parameter controlling randomness in output; low temperature makes outputs predictable, high makes them creative.",
        "type": "definition",
        "difficulty": "medium",
        "tags": ["temperature", "non-determinism"]
    },
    {
        "test_case_id": "tc005",
        "input": "What is the Oracle Problem in AI testing?",
        "expected_output": "The Oracle Problem is the challenge in AI testing where there is no single correct expected output for most queries; multiple valid responses exist.",
        "type": "definition",
        "difficulty": "hard",
        "tags": ["testing", "oracle"]
    }
]

# Function to ingest files (optional, if needed to load paste.txt content into RAG)
def ingest_files(file_paths):
    url = f"{RAG_BASE_URL}/ingest"
    with open(file_paths[0], 'rb') as f:  # Assuming paste.txt is local
        files = {'files': f}
        response = requests.post(url, files=files)
    print(f"Ingest response: {response.json()}")

# Uncomment if you have paste.txt locally and want to ingest
# ingest_files(["paste.txt"])

# Hit RAG /retrieve for each test case and fill actual_output + retrieval_context
for test_case in golden_dataset:
    retrieve_url = f"{RAG_BASE_URL}/retrieve"
    payload = {"query": test_case["input"], "top_k": 3}
    response = requests.post(retrieve_url, json=payload)
    if response.status_code == 200:
        rag_result = response.json()
        # Assuming response has 'answer' and 'context' fields (common in RAG APIs; adjust if different)
        test_case["actual_output"] = rag_result['answer']['answer']
        test_case["retrieval_context"] = normalize_retrieval_context(
            rag_result.get("answer", {}).get("context")
        )
        print(f"Filled tc{test_case['test_case_id']}: {test_case['actual_output'][:100]}...")
    else:
        print(f"Error for {test_case['input']}: {response.text}")
        test_case["actual_output"] = "API Error"
        test_case["retrieval_context"] = []

print("\nGolden Dataset with actual_outputs filled:")
print(json.dumps(golden_dataset, indent=2))

# DeepEval metrics configuration (uses AzureOpenAI)
answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=client,)
faithfulness = FaithfulnessMetric(threshold=0.7, model=client,)
context_recall = ContextualRecallMetric(threshold=0.7, model=client,)
context_precision = ContextualPrecisionMetric(threshold=0.7, model=client,)

# Evaluate each test case
test_cases = []
for item in golden_dataset:
    print(item["retrieval_context"])
    test_case = LLMTestCase(
        input=item["input"],
        actual_output=item["actual_output"],
        expected_output=item["expected_output"],
        retrieval_context=item["retrieval_context"]
    )
    test_cases.append(test_case)

# Run evaluation
print("\nRunning DeepEval...")
evaluation_result = evaluate(
    test_cases=test_cases,
    metrics=[answer_relevancy, faithfulness, context_recall, context_precision]
)

