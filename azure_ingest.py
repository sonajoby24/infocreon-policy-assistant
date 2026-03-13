from azure.storage.fileshare import ShareServiceClient
from dotenv import load_dotenv
import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------------------------
# Load ENV variables
# -------------------------

load_dotenv()

ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT")
ACCOUNT_KEY = os.getenv("AZURE_STORAGE_KEY")
SHARE_NAME = os.getenv("AZURE_FILE_SHARE_NAME")

connection_string = f"DefaultEndpointsProtocol=https;AccountName={ACCOUNT_NAME};AccountKey={ACCOUNT_KEY};EndpointSuffix=core.windows.net"

# -------------------------
# Azure connection
# -------------------------

service = ShareServiceClient.from_connection_string(connection_string)
share_client = service.get_share_client(SHARE_NAME)

TEMP_FOLDER = "temp_docs"
os.makedirs(TEMP_FOLDER, exist_ok=True)

# -------------------------
# Embedding Model
# -------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
metadata = []

# -------------------------
# Process PDF
# -------------------------

def process_pdf(file_path, filename):

    reader = PdfReader(file_path)

    for page_number, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:
            documents.append(text)

            metadata.append({
                "file": filename,
                "page": page_number + 1
            })

# -------------------------
# Azure Ingestion
# -------------------------

print("Starting Azure ingestion...")

files = share_client.list_directories_and_files()

for item in files:

    if item.is_directory:
        continue

    filename = item.name
    local_path = os.path.join(TEMP_FOLDER, filename)

    print("Downloading:", filename)

    file_client = share_client.get_file_client(filename)

    with open(local_path, "wb") as f:
        data = file_client.download_file()
        f.write(data.readall())

    process_pdf(local_path, filename)

    os.remove(local_path)

    print("Processed:", filename)

# -------------------------
# Create embeddings
# -------------------------

print("Creating embeddings...")

embeddings = embedding_model.encode(documents)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# -------------------------
# Save Vector DB
# -------------------------

faiss.write_index(index, "faiss_index.index")

np.save("documents.npy", documents)
np.save("metadata.npy", metadata)

print("Vector DB created successfully")