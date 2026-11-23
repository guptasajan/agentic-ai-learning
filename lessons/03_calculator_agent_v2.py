
from ollama import chat

# 1. Calculator tool
# def calculator(a: float, b: float, operation: str):
#     if operation == "add":
#         return a + b

#     elif operation == "subtract":
#         return a - b

#     elif operation == "multiply":
#         return a * b

#     elif operation == "divide":
#         if b == 0:
#             return "Cannot divide by zero"
#         return a / b

#     return "Invalid operation"

def calculator(a: float, b: float, operation: str):
    try:
        if operation == "add":
            return a + b

        elif operation == "subtract":
            return a - b

        elif operation == "multiply":
            return a * b

        elif operation == "divide":
            if b == 0:
                return "Error: Cannot divide by zero."

            return a / b

        else:
            return "Error: Unsupported operation."

    except Exception as e:
        return f"Calculator failed: {str(e)}"

# 2. Available tools
available_tools = {
    "calculator": calculator
}

# 3. User query
messages = [
    {
        "role": "user",
        "content": "Calculate 25 divided by 0, then add 50 to the result."
    }
]

# 4. Agent loop
max_iterations = 5

for iteration in range(max_iterations):

    print(f"\nAgent iteration: {iteration + 1}")

    response = chat(
        model="qwen3:1.7b",
        messages=messages,
        tools=[calculator],
        think=True
    )

    messages.append(response.message)

    # Check tool calls
    if not response.message.tool_calls:

        print("\nFinal Answer:")
        print(response.message.content)
        break

    # Execute requested tools
    for tool_call in response.message.tool_calls:

        tool_name = tool_call.function.name
        arguments = tool_call.function.arguments

        print("Selected tool:", tool_name)
        print("Arguments:", arguments)

        function = available_tools.get(tool_name)

        if function is None:
            result = "Unknown tool"
        else:
            result = function(**arguments)

        print("Tool result:", result)

        # Send result back to LLM
        messages.append({
            "role": "tool",
            "tool_name": tool_name,
            "content": str(result)
        })

else:
    print("\nMaximum iterations reached.")

