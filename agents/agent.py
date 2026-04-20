import os
import json
from dotenv import load_dotenv

load_dotenv()

from openai import AzureOpenAI
from deepeval.tracing import observe

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")


@observe(type="tool", name="weather_tool")
def get_weather(city: str):
    return f"The current weather in {city} is 30°C and sunny."


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }
]


@observe(type="agent", name="simple_azure_agent")
def run_agent(user_query: str):
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": user_query}
    ]

    tools_used = []

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto"
    )

    message = response.choices[0].message

    if message.tool_calls:
        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)

            tool_result = get_weather(args["city"])

            tools_used.append({
                "name": "get_weather",
                "arguments": args
            })

            messages.append(message)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

        final_response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=messages
        )

        return {
            "output": final_response.choices[0].message.content,
            "tools_called": tools_used
        }

    return {
        "output": message.content,
        "tools_called": []
    }

if __name__ == "__main__":
    result = run_agent("What is weather in Hyderabad?")
    print(result)