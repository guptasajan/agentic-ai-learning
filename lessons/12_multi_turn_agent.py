
import chromadb
import ollama

# ==================================
# STEP 1: CONNECT TO CHROMADB
# ==================================
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="agent_memory"
)


# ==================================
# STEP 2: EMBEDDING FUNCTION
# ==================================
def generate_embedding(text):

    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )

    return response["embedding"]


# ==================================
# STEP 3: MEMORY SEARCH TOOL
# ==================================

def memory_search(query):

    query_embedding = generate_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    return results["documents"][0]


# ==================================
# STEP 4: CALCULATOR TOOL
# ==================================

def calculator(a: float, b: float, operation: str):

    if operation == "add":
        return a + b

    elif operation == "subtract":
        return a - b

    elif operation == "multiply":
        return a * b

    elif operation == "divide":

        if b == 0:
            return "Error: Division by zero."

        return a / b

    return "Invalid operation."


# ==================================
# STEP 5: TOOL REGISTRY
# ==================================

available_tools = {
    "memory_search": memory_search,
    "calculator": calculator
}


# ==================================
# STEP 6: TOOL DEFINITIONS
# ==================================

tools = [

    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search stored user memories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    }
                },
                "required": ["query"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform mathematical calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "operation": {
                        "type": "string",
                        "enum": [
                            "add",
                            "subtract",
                            "multiply",
                            "divide"
                        ]
                    }
                },
                "required": ["a", "b", "operation"]
            }
        }
    }

]


# ==================================
# STEP 7: CONVERSATION MEMORY
# ==================================

messages = [

    {
        "role": "system",
        "content": (
            "You are an intelligent AI assistant. "
            "Remember the current conversation. "
            "Use memory_search for stored personal information. "
            "Use calculator for mathematical calculations. "
            "Never invent missing information."
        )
    }

]


# ==================================
# STEP 8: AGENT FUNCTION
# ==================================

def run_agent(user_query):

    # Add new user message to existing history
    messages.append({
        "role": "user",
        "content": user_query
    })

    max_iterations = 5

    for iteration in range(max_iterations):

        print(f"\nAgent iteration: {iteration + 1}")

        response = ollama.chat(
            model="qwen3:1.7b",
            messages=messages,
            tools=tools,
            think=True
        )

        # Save assistant response
        messages.append(response.message)

        # If no tool call, agent has generated final answer
        if not response.message.tool_calls:

            print("\nAgent:", response.message.content)
            return

        # Execute requested tools
        for tool_call in response.message.tool_calls:

            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments

            print("Selected tool:", tool_name)

            function_to_call = available_tools.get(tool_name)

            if function_to_call is None:
                result = "Error: Unknown tool."

            else:
                try:
                    result = function_to_call(**arguments)

                except Exception as error:
                    result = f"Tool error: {error}"

            print("Tool result:", result)

            # Send tool result back to model
            messages.append({
                "role": "tool",
                "name": tool_name,
                "content": str(result)
            })

    print("Agent stopped: Maximum iterations reached.")


# ==================================
# STEP 9: CHAT LOOP
# ==================================

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Agent: Goodbye!")
        break

    run_agent(user_input)
