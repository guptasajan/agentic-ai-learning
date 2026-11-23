from ollama import chat
import json
import os
import re

# MEMORY FILE
MEMORY_FILE = "memory.json"


# LOAD MEMORY
if os.path.exists(MEMORY_FILE):

    with open(MEMORY_FILE, "r", encoding="utf-8") as file:
        memory = json.load(file)

else:
    memory = []


# COMMON WORDS TO IGNORE
stop_words = {
    "what", "is", "my", "i", "am", "the",
    "a", "an", "do", "you", "me", "at",
    "and", "in", "to", "of", "it", "are",
    "which", "where", "how"
}


# MEMORY RETRIEVAL FUNCTION
def retrieve_memory(query):

    query_words = set(
        re.findall(r"\b\w+\b", query.lower())
    )

    query_words = query_words - stop_words

    scored_memories = []

    for index, message in enumerate(memory):

        content = message.get("content", "")

        memory_words = set(
            re.findall(r"\b\w+\b", content.lower())
        )

        memory_words = memory_words - stop_words

        score = len(query_words.intersection(memory_words))

        if score > 0:
            scored_memories.append((score, index, message))

    # Sort by relevance score
    scored_memories.sort(
        key=lambda item: item[0],
        reverse=True
    )

    # Select top 4 relevant memories
    selected = scored_memories[:4]

    # Restore original conversation order
    selected.sort(key=lambda item: item[1])

    return [item[2] for item in selected]


# SAVE MEMORY FUNCTION
def save_memory():

    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(memory, file, indent=4, ensure_ascii=False)


# AGENT START
print("Smart Memory Agent Started!")
print("Type 'exit' to stop.\n")


while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Agent: Goodbye!")
        break

    # RETRIEVE RELEVANT MEMORY
    relevant_memory = retrieve_memory(user_input)

    print("\nRetrieved Memory:")
    print(relevant_memory)

    # BUILD CONTEXT FOR LLM
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant. "
                "Use the provided conversation history as memory. "
                "Answer using relevant information from that history. "
                "If the answer is not present, say you do not know."
            )
        }
    ]

    messages.extend(relevant_memory)

    messages.append({
        "role": "user",
        "content": user_input
    })

    # CALL LLM
    response = chat(
        model="qwen3:1.7b",
        messages=messages,
        think=True
    )

    print("\nAgent:", response.message.content)

    # SAVE NEW CONVERSATION TO LONG-TERM MEMORY
    memory.append({
        "role": "user",
        "content": user_input
    })

    memory.append({
        "role": "assistant",
        "content": response.message.content
    })

    save_memory()

    print()
