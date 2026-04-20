import os
from dotenv import load_dotenv

load_dotenv()

from deepeval import assert_test
from deepeval.metrics import (
    TaskCompletionMetric,
    ToolCorrectnessMetric,
    ArgumentCorrectnessMetric,
    ConversationCompletenessMetric,
    TurnRelevancyMetric
)
from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    Turn,
    ToolCall
)
from deepeval.models import AzureOpenAIModel

from agent import run_agent

judge = AzureOpenAIModel(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)


# ---------------- SINGLE TURN TEST ---------------- #
def test_single_turn_agent_trace():
    query = "What is weather in Hyderabad?"

    result = run_agent(query)

    tools_called = [
        ToolCall(
            name=tool["name"],
            input_parameters=tool["arguments"],
            output=result["output"]
        )
        for tool in result["tools_called"]
    ]

    test_case = LLMTestCase(
        input=query,
        actual_output=result["output"],
        tools_called=tools_called,
        expected_tools=[
            ToolCall(
                name="get_weather",
                input_parameters={"city": "Hyderabad"},
            )
        ],
    )

    metrics = [
        TaskCompletionMetric(threshold=0.7, model=judge),
        ToolCorrectnessMetric(threshold=0.7, model=judge),
        ArgumentCorrectnessMetric(threshold=0.7, model=judge)
    ]

    for metric in metrics:
        assert_test(test_case, [metric])


# ---------------- MULTI TURN TEST ---------------- #
def test_multi_turn_trace():
    result1 = run_agent("What is weather in Hyderabad?")
    turn2 = "Tomorrow it will be cloudy"

    conversation = ConversationalTestCase(
        turns=[
            Turn(role="user", content="What is weather in Hyderabad?"),
            Turn(role="assistant", content=result1["output"]),
            Turn(role="user", content="What about tomorrow?"),
            Turn(role="assistant", content=turn2),
        ]
    )

    metrics = [
        ConversationCompletenessMetric(
            threshold=0.7,
            model=judge
        ),
        TurnRelevancyMetric(
            threshold=0.7,
            model=judge
        )
    ]

    for metric in metrics:
        metric.measure(conversation)
        print(f"{metric.__class__.__name__}: {metric.score}")