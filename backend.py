"""
FastAPI Backend for Infocreon Policy Assistant

This backend server connects the chatbot frontend with the RAG engine.

Workflow:
1. Receive user question from frontend (React chatbot)
2. Send the question to the RAG engine
3. Retrieve the generated answer from the LLM
4. Return the answer along with the source file and page number

Technologies Used:
- FastAPI for backend API server
- RAG Engine for document retrieval and answer generation
"""
# Import FastAPI framework
from fastapi import FastAPI
# Import CORS middleware to allow frontend requests
from fastapi.middleware.cors import CORSMiddleware
# Import Pydantic model for request validation
from pydantic import BaseModel
# Import RAG query function
from rag_engine_azure import query_rag_azure

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

# API endpoint that receives user questions
@app.post("/ask")
async def ask_question(data: Question):
    # Send question to RAG engine
    result = query_rag_azure(data.question)
    # Return answer + source citation to frontend
    return {
        "answer": result["answer"],
        "source": result["source"],
        "page": result["page"],
        "ranking": result["ranking"]   # send ranking to UI
    }