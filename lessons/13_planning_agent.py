
import ollama
import json


def create_plan(user_task):

    
    prompt = f"""
    You are an AI task planning assistant.

    Break the user's complete task into ALL necessary
    steps. Do not skip any requested action.

    Available tools:
    1. memory_search: Search stored user memories.
    2. calculator: Perform mathematical calculations.
    3. none: Used when no tool is needed.

    Rules:
    - Every separate user request must be covered.
    - Use memory_search for personal information.
    - Use calculator for mathematical operations.
    - Add a final step to combine or summarize results
    whenever the user asks for a combined answer.
    - Do not stop after the first step.
    - Create at least one step for every requested action.

    Return valid JSON in this exact structure:

    {{
        "task": "Short task description",
        "steps": [
            {{
                "step": 1,
                "action": "Description of action",
                "tool": "memory_search"
            }},
            {{
                "step": 2,
                "action": "Description of action",
                "tool": "calculator"
            }},
            {{
                "step": 3,
                "action": "Combine results",
                "tool": "none"
            }}
        ]
    }}




    User Task: {user_task}
    """

    response = ollama.chat(
        model="qwen3:1.7b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json",
        think=True
    )

    plan = json.loads(response.message.content)

    return plan


# User interaction
while True:

    user_task = input("\nYou: ")

    if user_task.lower() == "exit":
        print("Planner: Goodbye!")
        break

    try:

        plan = create_plan(user_task)

        print("\nGenerated Plan:\n")

        print("Task:", plan["task"])

        for step in plan["steps"]:

            print(
                f"\nStep {step['step']}: "
                f"{step['action']}"
            )

            print("Tool:", step["tool"])

    except (ValueError, KeyError, TypeError) as error:

        print("Could not process the generated plan:", error)
