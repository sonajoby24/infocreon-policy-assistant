"""
RAG Engine for Infocreon Policy Assistant

This module handles user queries by performing Retrieval-Augmented Generation (RAG).

Steps:
1. Convert user question into vector embedding
2. Retrieve most relevant document chunks from FAISS
3. Rank retrieved results based on similarity score
4. Send context to Gemini LLM
5. Generate final answer with source citation
"""
import faiss
import os
from dotenv import load_dotenv
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Load Azure storage credentials from environment variables
load_dotenv()
# -------------------------
# Gemini Configuration
# -------------------------
# Configure Google Gemini API for LLM responses
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")


# -------------------------
# Embedding Model
# -------------------------
# Load embedding model used for converting text into vectors
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------
# Load Azure Vector Database
# -------------------------
# Load FAISS vector database containing document embeddings
index = faiss.read_index("faiss_index.index")

# Load stored document chunks
documents = np.load("documents.npy", allow_pickle=True)

# Load metadata containing source file name and page number
metadata = np.load("metadata.npy", allow_pickle=True)


# -------------------------
# Query Function
# -------------------------
# Main function that processes user query using RAG pipeline
def query_rag_azure(question):

    print("Embedding question...")

    # Convert user question into embedding vector
    q_embed = embed_model.encode([question])

    # Retrieve top 5 most similar document chunks from FAISS
    D, I = index.search(np.array(q_embed), 5)

    # ---------------------------------------------------------
    # 1️⃣ Rank retrieved files based on similarity score
    # ---------------------------------------------------------
    # Combine similarity scores and document indexes
    results = list(zip(D[0], I[0]))
    
    # Rank retrieved documents based on similarity score
    results = sorted(results, key=lambda x: x[0], reverse=True)

    print("\nRanked Results Based on Similarity Score:\n")

    ranked_docs = []
    ranking_output = []

    for rank, (score, idx) in enumerate(results):

        file = metadata[idx]["file"]
        page = metadata[idx]["page"]
        text = documents[idx]

        ranked_docs.append(text)

        print(f"Rank {rank+1} | Score: {score} | File: {file} | Page: {page}")

        ranking_output.append({
            "rank": rank + 1,
            "score": round(float(score), 3),
            "file": file,
            "page": page
        })

    context = "\n\n".join(ranked_docs)

    source = metadata[results[0][1]]["file"]
    page = metadata[results[0][1]]["page"]

    prompt = f"""
You are an HR policy assistant.

Answer ONLY using the policy text below.

If the answer is not found say:
"I could not find this in the policy documents."

Policy Text:
{context}

Question:
{question}
"""

    # ---------------------------------------------------------
    # 2️⃣ Set temperature = 0
    # ---------------------------------------------------------

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0
        }
    )

    answer = response.text if hasattr(response, "text") else str(response)

    return {
        "answer": answer,
        "source": source,
        "page": page,
        "ranking": ranking_output
    }