
def calculator(a, b, operation):
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    elif operation == "substract":
            return a - b
    elif operation == "divide":
            return a / b

user_goal = "Multiply 25 and 4"

print ("User Goal:", user_goal)

print("Agent: Selecting calculator tool...")

result = calculator(25, 4, "multiply")

print ("Agent: Task completed!")
print ("Final Answer:", result)