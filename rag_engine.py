import os
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import google.generativeai as genai
from google.generativeai import GenerativeModel


# -----------------------------------
# Load ENV variables
# -----------------------------------

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

model = GenerativeModel("gemini-2.5-flash")


# -----------------------------------
# Config
# -----------------------------------

DOCS_FOLDER = "docs"
INDEX_FOLDER = "faiss_index"

embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")


# -----------------------------------
# Build FAISS Index
# -----------------------------------

def build_index():

    docs = []

    print("Loading documents...")

    for file in os.listdir(DOCS_FOLDER):

        if file.endswith(".pdf"):

            loader = PyPDFLoader(f"{DOCS_FOLDER}/{file}")

            pages = loader.load()

            for p in pages:

                p.metadata["source"] = file

                docs.append(p)

    print("Splitting documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    texts = [c.page_content for c in chunks]

    print("Creating embeddings...")

    embeddings = embed_model.encode(texts)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)

    index.add(np.array(embeddings))

    os.makedirs(INDEX_FOLDER, exist_ok=True)

    print("Saving vector database...")

    faiss.write_index(index, f"{INDEX_FOLDER}/index.faiss")

    with open(f"{INDEX_FOLDER}/meta.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("Index built successfully")


# -----------------------------------
# Load Index
# -----------------------------------

def load_index():

    index = faiss.read_index(f"{INDEX_FOLDER}/index.faiss")

    with open(f"{INDEX_FOLDER}/meta.pkl", "rb") as f:
        docs = pickle.load(f)

    return index, docs


# -----------------------------------
# Query RAG
# -----------------------------------

def query_rag(question):

    index, docs = load_index()

    print("Embedding question...")

    q_embed = embed_model.encode([question])

    D, I = index.search(np.array(q_embed), 5)

    results = [docs[i] for i in I[0]]

    # Use top 3 chunks for better context
    context = "\n\n".join([doc.page_content for doc in results[:3]])

    source = results[0].metadata["source"]

    page = results[0].metadata.get("page", 1) + 1


    prompt = f"""
You are an HR policy assistant.

Answer ONLY using the policy text below.

If the answer is not in the document say:
"I could not find this in the policy documents."

Policy Text:
{context}

Question:
{question}
"""


    response = model.generate_content(prompt)

    answer = response.text if hasattr(response, "text") else str(response)

    return {
        "answer": answer,
        "source": source,
        "page": page
    }