
from ollama import chat

# Conversation history = Short-term memory
messages = []

print("Memory Agent Started!")
print("Type 'exit' to stop.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Agent: Goodbye!")
        break

    # Save user message
    messages.append({
        "role": "user",
        "content": user_input
    })

    # Send complete conversation history
    response = chat(
        model="qwen3:1.7b",
        messages=messages,
        think=True
    )

    # Save assistant response
    messages.append(response.message)

    print("\nAgent:", response.message.content)
    print()
