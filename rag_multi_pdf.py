import os
import pickle
import faiss

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

DOCS_FOLDER = "docs"
INDEX_FOLDER = "faiss_index"

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_or_create_index():

    if os.path.exists(f"{INDEX_FOLDER}/index.faiss"):

        index = faiss.read_index(f"{INDEX_FOLDER}/index.faiss")

        with open(f"{INDEX_FOLDER}/meta.pkl","rb") as f:
            texts = pickle.load(f)

        return index, texts

    return build_index()


def build_index():

    texts = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    for file in os.listdir(DOCS_FOLDER):

        if file.endswith(".pdf"):

            loader = PyPDFLoader(f"{DOCS_FOLDER}/{file}")
            pages = loader.load()

            chunks = splitter.split_documents(pages)

            for c in chunks:

                texts.append({
                    "text": c.page_content,
                    "source": file,
                    "page": c.metadata["page"]
                })

    embeddings = model.encode([t["text"] for t in texts])

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    os.makedirs(INDEX_FOLDER, exist_ok=True)

    faiss.write_index(index, f"{INDEX_FOLDER}/index.faiss")

    with open(f"{INDEX_FOLDER}/meta.pkl","wb") as f:
        pickle.dump(texts,f)

    return index, texts


def query_rag(question):

    index, texts = load_or_create_index()

    q_embedding = model.encode([question])

    D,I = index.search(q_embedding,5)

    best = texts[I[0][0]]

    answer = best["text"]

    source = f"{best['source']} (Page {best['page']})"

    return answer, source