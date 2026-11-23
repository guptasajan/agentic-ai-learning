
from ollama import chat
import json
import os

# MEMORY FILE
MEMORY_FILE = "memory.json"


# LOAD PREVIOUS MEMORY
if os.path.exists(MEMORY_FILE):

    with open(MEMORY_FILE, "r", encoding="utf-8") as file:
        messages = json.load(file)

    print("Previous memory loaded!")

else:
    messages = []
    print("No previous memory found. Starting fresh!")


# SAVE MEMORY FUNCTION
def save_memory():
    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(messages, file, indent=4, ensure_ascii=False)


# AGENT START
print("\nPersistent Memory Agent Started!")
print("Type 'exit' to stop.\n")


while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        save_memory()
        print("Agent: Memory saved. Goodbye!")
        break

    # SAVE USER MESSAGE
    messages.append({
        "role": "user",
        "content": user_input
    })

    # SEND CONVERSATION TO LLM
    response = chat(
        model="qwen3:1.7b",
        messages=messages,
        think=True
    )

    # SAVE ASSISTANT RESPONSE
    messages.append({
        "role": "assistant",
        "content": response.message.content
    })

    print("\nAgent:", response.message.content)

    # SAVE UPDATED MEMORY
    save_memory()

    print()
