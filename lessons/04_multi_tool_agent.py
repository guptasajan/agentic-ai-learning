
from ollama import chat

# TOOL 1: Calculator
def calculator(a: float, b: float, operation: str):
    """Perform mathematical calculations."""

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


# TOOL 2: Text Analyzer
def text_analyzer(text: str):
    """Analyze text and return word and character counts."""

    words = text.split()

    return {
        "word_count": len(words),
        "character_count": len(text)
    }


# REGISTER TOOLS
available_tools = {
    "calculator": calculator,
    "text_analyzer": text_analyzer
}


# USER QUERY
messages = [
    {
        "role": "user",
        "content": (
            "Calculate 25 divided by 0. "
            "Also analyze the text 'Agentic AI is powerful' "
            "and tell me its word count."
        )
    }
]


# AGENT LOOP
for iteration in range(5):

    print(f"\nAgent iteration: {iteration + 1}")

    response = chat(
        model="qwen3:1.7b",
        messages=messages,
        tools=[calculator, text_analyzer],
        think=True
    )

    messages.append(response.message)

    if not response.message.tool_calls:

        print("\nFinal Answer:")
        print(response.message.content)
        break

    for tool_call in response.message.tool_calls:

        tool_name = tool_call.function.name
        arguments = tool_call.function.arguments

        print("\nSelected tool:", tool_name)
        print("Arguments:", arguments)

        function = available_tools.get(tool_name)

        try:
            if function:
                result = function(**arguments)
            else:
                result = f"Error: Unknown tool '{tool_name}'"

        except Exception as e:
            result = f"Tool execution failed: {str(e)}"

        print("Tool result:", result)

        messages.append({
            "role": "tool",
            "tool_name": tool_name,
            "content": str(result)
        })

else:
    print("Maximum iterations reached.")
