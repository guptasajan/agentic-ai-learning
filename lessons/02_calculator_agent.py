
from ollama import chat

# 1. Define calculator tool
def calculator(a: float, b: float, operation: str) -> float:
    """
    Perform mathematical calculations.

    Args:
        a: First number
        b: Second number
        operation: add, subtract, multiply or divide
    """

    if operation == "add":
        return a + b

    elif operation == "multiply":
        return a * b

    elif operation == "subtract":
        return a - b

    elif operation == "divide":
        if b == 0:
            return "Cannot divide by zero"
        return a / b

    return "Invalid operation"


# 2. Register available tools
available_tools = {
    "calculator": calculator
}


# 3. User query
messages = [
    {
        "role": "user",
        "content": "What is 25 multiplied by 4?"
    }
]

print("User: What is 25 multiplied by 4?")


# 4. Send query to LLM
response = chat(
    model="qwen3:1.7b",
    messages=messages,
    tools=[calculator],
    think=True
)


# 5. Check tool calls
messages.append(response.message)

if response.message.tool_calls:

    for tool_call in response.message.tool_calls:

        tool_name = tool_call.function.name
        arguments = tool_call.function.arguments

        print("\nAgent selected tool:", tool_name)
        print("Arguments:", arguments)

        # 6. Execute selected tool
        function = available_tools.get(tool_name)

        if function:
            result = function(**arguments)
        else:
            result = "Unknown tool"

        print("Tool execution result:", result)

        # 7. Send result back to LLM
        messages.append({
            "role": "tool",
            "tool_name": tool_name,
            "content": str(result)
        })

    # 8. Generate final response
    final_response = chat(
        model="qwen3:1.7b",
        messages=messages,
        think=True
    )

    print("\nAgent Final Answer:")
    print(final_response.message.content)

else:
    print("\nAgent Response:")
    print(response.message.content)
