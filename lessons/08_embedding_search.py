
import ollama
import math

# EMBEDDING MODEL
EMBED_MODEL = "nomic-embed-text"


# SAMPLE MEMORIES
memories = [
    "I am learning Agentic AI.",
    "I study at NSUT.",
    "I use a MacBook Air M1.",
    "I enjoy solving programming problems.",
    "I am interested in software development."
]


# FUNCTION TO GENERATE EMBEDDING
def generate_embedding(text):

    response = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text
    )

    return response["embedding"]


# COSINE SIMILARITY FUNCTION
def cosine_similarity(vector_a, vector_b):

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product / (magnitude_a * magnitude_b)


# GENERATE MEMORY VECTORS
memory_vectors = []

for memory in memories:

    vector = generate_embedding(memory)

    memory_vectors.append({
        "text": memory,
        "embedding": vector
    })


# SEARCH FUNCTION
def semantic_search(query):

    query_vector = generate_embedding(query)

    results = []

    for item in memory_vectors:

        score = cosine_similarity(
            query_vector,
            item["embedding"]
        )

        results.append({
            "text": item["text"],
            "score": score
        })

    # Highest similarity first
    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:3]


# TEST QUERY
# query = "What is my current area of study?"
query = "Which university am I studying at?"
query = "What kind of programming work do I enjoy?"


results = semantic_search(query)

print("\nSearch Query:", query)
print("\nTop Relevant Memories:\n")

for result in results:

    print("Memory:", result["text"])
    print("Similarity:", round(result["score"], 4))
    print()
