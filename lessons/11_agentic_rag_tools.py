
import chromadb
import ollama


# =====================================
# STEP 1: CONNECT TO CHROMADB
# =====================================

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="agent_memory"
)


# =====================================
# STEP 2: GENERATE EMBEDDINGS
# =====================================

def generate_embedding(text):

    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )

    return response["embedding"]


# =====================================
# STEP 3: ADD SAMPLE MEMORIES
# =====================================

memories = [
    "I am learning Agentic AI.",
    "I study at NSUT.",
    "I use a MacBook Air M1.",
    "I enjoy solving programming problems.",
    "I am interested in software development."
]

ids = []
documents = []
embeddings = []

for i, memory in enumerate(memories):

    ids.append(f"memory_{i + 1}")
    documents.append(memory)
    embeddings.append(generate_embedding(memory))

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)


# =====================================
# STEP 4: MEMORY SEARCH TOOL
# =====================================

def memory_search(query: str):

    query_embedding = generate_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    return results["documents"][0]


# =====================================
# STEP 5: CALCULATOR TOOL
# =====================================

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


# =====================================
# STEP 6: DEFINE AVAILABLE TOOLS
# =====================================

available_tools = {

    "memory_search": memory_search,
    "calculator": calculator

}


# =====================================
# STEP 7: TOOL DEFINITIONS FOR QWEN
# =====================================

tools = [

    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search the user's stored memories using semantic search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Information to search for in memory."
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
                        "enum": ["add", "subtract", "multiply", "divide"]
                    }
                },
                "required": ["a", "b", "operation"]
            }
        }
    }

]


# =====================================
# STEP 8: AGENT LOOP
# =====================================

def run_agent(user_query):

    messages = [
        {
            "role": "system",
            "content": (
                "You are an intelligent AI agent. "
                "Use memory_search when the user asks about "
                "their stored personal information. "
                "Use calculator for mathematical calculations. "
                "Use tool results to formulate your final answer. "
                "Do not invent missing personal information."
            )
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    max_iterations = 5

    for iteration in range(max_iterations):

        print(f"\nAgent iteration: {iteration + 1}")

        response = ollama.chat(
            model="qwen3:1.7b",
            messages=messages,
            tools=tools,
            think=True
        )

        # Add assistant response to conversation
        messages.append(response.message)

        # Check whether the model requested a tool
        if not response.message.tool_calls:

            print("\nAgent:", response.message.content)
            return

        # Execute requested tools
        for tool_call in response.message.tool_calls:

            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments

            print("Selected tool:", tool_name)
            print("Arguments:", arguments)

            function_to_call = available_tools.get(tool_name)

            if function_to_call is None:
                result = "Error: Unknown tool."
            else:
                try:
                    result = function_to_call(**arguments)

                except Exception as error:
                    result = f"Tool error: {error}"

            print("Tool result:", result)

            # Send tool result back to the model
            messages.append({
                "role": "tool",
                "name": tool_name,
                "content": str(result)
            })

    print("Agent stopped: Maximum iterations reached.")


# =====================================
# STEP 9: USER INTERACTION
# =====================================

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Agent: Goodbye!")
        break

    run_agent(user_input)
