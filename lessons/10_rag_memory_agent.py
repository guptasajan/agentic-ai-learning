
import chromadb
import ollama

# STEP 1: Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="agent_memory"
)


# STEP 2: Generate embeddings
def generate_embedding(text):

    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )

    return response["embedding"]


# STEP 3: Add memories to database
memories = [
    "I am learning Agentic AI.",
    "I study at NSUT.",
    "I use a MacBook Air M1.",
    "I enjoy solving programming problems.",
    "I am interested in software development."
]

for i, memory in enumerate(memories):

    collection.upsert(
        ids=[f"memory_{i + 1}"],
        documents=[memory],
        embeddings=[generate_embedding(memory)]
    )


# STEP 4: Retrieve relevant memories
def retrieve_memories(query):

    query_embedding = generate_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    return results["documents"][0]


# STEP 5: Generate answer using retrieved memories
def ask_agent(user_query):

    relevant_memories = retrieve_memories(user_query)

    context = "\n".join(relevant_memories)

    prompt = f"""
    Answer the user's question using the following memories.

    Relevant Memories:
    {context}

    User Question:
    {user_query}

    If the answer is not available in the memories,
    clearly say that you don't have that information.
    """

    response = ollama.chat(
        model="qwen3:1.7b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        think=True
    )

    return response.message.content


# STEP 6: Chat loop
while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Agent: Goodbye!")
        break

    answer = ask_agent(user_input)

    print("\nAgent:", answer)
