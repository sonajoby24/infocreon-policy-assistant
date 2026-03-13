"""
Azure Ingestion Pipeline

This script downloads PDF policy documents from Azure File Share,
extracts text from each page, generates embeddings using
SentenceTransformer, and stores them in a FAISS vector database.

Workflow:
Azure File Share → Download PDFs → Extract Text → Generate Embeddings → FAISS Index
"""

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
# Load Azure storage credentials from environment variables
load_dotenv()

ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT")
ACCOUNT_KEY = os.getenv("AZURE_STORAGE_KEY")
SHARE_NAME = os.getenv("AZURE_FILE_SHARE_NAME")
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
AZURE_FILE_SHARE_NAME = os.getenv("AZURE_FILE_SHARE_NAME")
AZURE_STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY")
AZURE_ROOT_FOLDER = os.getenv("AZURE_ROOT_FOLDER")

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
# Load embedding model to convert text into vector representations
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Lists to store extracted document text and metadata
documents = []
metadata = []

# -------------------------
# Process PDF
# -------------------------
# Extract text from each page of the PDF and store metadata
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
# Recursive directory scan
# -------------------------
# Recursively scan Azure File Share directories for PDF files
def process_directory(directory=""):

    dir_client = share_client.get_directory_client(directory)

    items = dir_client.list_directories_and_files()

    for item in items:

        if item.is_directory:

            # Go inside subfolder
            process_directory(os.path.join(directory, item.name))

        else:

            filename = item.name

            if not filename.endswith(".pdf"):
                continue

            azure_path = os.path.join(directory, filename)

            local_path = os.path.join(TEMP_FOLDER, filename)

            print("Downloading:", azure_path)

            file_client = share_client.get_file_client(azure_path)

            with open(local_path, "wb") as f:
                data = file_client.download_file()
                f.write(data.readall())

            process_pdf(local_path, filename)

            os.remove(local_path)

            print("Processed:", filename)


# -------------------------
# Start ingestion
# -------------------------

print("Starting Azure ingestion...")

process_directory()

# -------------------------
# Create embeddings
# -------------------------
# Convert extracted document text into vector embeddings
print("Creating embeddings...")

embeddings = embedding_model.encode(documents)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

# -------------------------
# Save Vector DB
# -------------------------
# Save FAISS vector database and document metadata
faiss.write_index(index, "faiss_index.index")

np.save("documents.npy", documents)

np.save("metadata.npy", metadata)

print("Vector DB created successfully")