# Infocreon Policy Assistant

A Retrieval-Augmented Generation (RAG) based AI assistant that allows users to query company policy documents.

## Tech Stack
- Python
- Azure OpenAI
- FAISS Vector Search
- SharePoint
- Azure Storage
- React

## Features
- Upload policy documents from SharePoint
- Create embeddings using Azure OpenAI
- Store vectors in FAISS
- Ask questions about company policies
- Chatbot interface built with React

## Project Structure

backend.py – API backend  
rag_engine.py – RAG logic  
rag_engine_azure.py – Azure OpenAI integration  
azure_ingest.py – document ingestion  
sharepoint_to_azure.py – SharePoint integration  
faiss_retriever.py – vector search  
frontend/ – React chatbot UI  

## Setup

Install dependencies

```
pip install -r requirements.txt
```

Run backend

```
python backend.py
```

Run frontend

```
npm start
```
