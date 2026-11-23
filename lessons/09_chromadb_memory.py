
import chromadb
import ollama

# Step 1: Connect to persistent ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# Step 2: Create or access a collection
collection = client.get_or_create_collection(
    name="agent_memory"
)

# Step 3: Our sample memories
memories = [
    "I am learning Agentic AI.",
    "I study at NSUT.",
    "I use a MacBook Air M1.",
    "I enjoy solving programming problems.",
    "I am interested in software development."
]

# Step 4: Generate embeddings using Ollama
def generate_embedding(text):
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    return response["embedding"]


# Step 5: Store memories and their embeddings
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

print("Memories saved in ChromaDB!")


# Step 6: Semantic search
def semantic_search(query):

    query_embedding = generate_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    return results


# Step 7: Test the search
query = "Which university am I studying at?"

results = semantic_search(query)

print("\nSearch Query:", query)
print("\nRelevant Memories:\n")

for memory in results["documents"][0]:
    print("-", memory)
