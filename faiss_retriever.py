import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

class FAISSRetriever:

    def __init__(self):

        self.docs_folder = "docs"
        self.index_folder = "faiss_index"

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        os.makedirs(self.index_folder, exist_ok=True)

        self.index_path = os.path.join(self.index_folder,"index.faiss")
        self.meta_path = os.path.join(self.index_folder,"meta.pkl")

        if os.path.exists(self.index_path):
            print("Loading existing FAISS index...")
            self.index = faiss.read_index(self.index_path)

            with open(self.meta_path,"rb") as f:
                self.metadata = pickle.load(f)

        else:
            print("Creating FAISS index first time...")
            self.metadata = []
            self.build_index()


    def build_index(self):

        texts = []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        for file in os.listdir(self.docs_folder):

            if file.endswith(".pdf"):

                path = os.path.join(self.docs_folder,file)

                loader = PyPDFLoader(path)
                pages = loader.load()

                chunks = splitter.split_documents(pages)

                for c in chunks:

                    texts.append(c.page_content)

                    self.metadata.append({
                        "text":c.page_content,
                        "source":file,
                        "page":c.metadata["page"]+1
                    })

        embeddings = self.model.encode(texts)

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(np.array(embeddings).astype("float32"))

        self.save()


    def save(self):

        faiss.write_index(self.index,self.index_path)

        with open(self.meta_path,"wb") as f:
            pickle.dump(self.metadata,f)


    def add_new_pdf(self,file_path):

        loader = PyPDFLoader(file_path)

        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(pages)

        texts = [c.page_content for c in chunks]

        embeddings = self.model.encode(texts)

        self.index.add(np.array(embeddings).astype("float32"))

        for c in chunks:

            self.metadata.append({
                "text":c.page_content,
                "source":os.path.basename(file_path),
                "page":c.metadata["page"]+1
            })

        self.save()


    def search(self,query,k=3):

        emb = self.model.encode([query])

        D,I = self.index.search(np.array(emb).astype("float32"),k)

        results=[]

        for idx in I[0]:
            results.append(self.metadata[idx])

        return results