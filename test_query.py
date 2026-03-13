import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load vector DB
index = faiss.read_index("faiss_index.index")

documents = np.load("documents.npy", allow_pickle=True)
metadata = np.load("metadata.npy", allow_pickle=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

while True:

    question = input("\nAsk question: ")

    query_vector = model.encode([question])

    D, I = index.search(np.array(query_vector), 3)

    for idx in I[0]:
        print("\nAnswer chunk:")
        print(documents[idx])
        print("Source:", metadata[idx])