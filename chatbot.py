import os
import argparse
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from pymongo import MongoClient
from gridfs import GridFS
from bson import ObjectId
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO

import fitz  # PyMuPDF
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient



# === ENV SETUP === #
load_dotenv()
HF_TOKEN = os.getenv("HF_API_KEY")
MONGO_URI = os.getenv("MONGODB_URI")
MONGO_DB = os.getenv("MONGODB_DATABASE")
CHROMA_DIR = "chroma_db"

# === DB & Model Clients === #
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token=HF_TOKEN)
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB]
fs = GridFS(db)

# === FastAPI Setup === #

origins = [
    "http://localhost:3000", 
    "http://localhost:3001",
    "http://localhost:3002",   
    "https://insurealltheway.co", 
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    user_id: str = None

# === PDF Embedding Pipeline === #
def extract_text_from_pdf_stream(pdf_stream, source_name):
    pdf_stream.seek(0)
    doc = fitz.open(stream=pdf_stream.read(), filetype="pdf")
    documents = []
    for i, page in enumerate(doc):
        text = page.get_text()
        metadata = {"source": f"{source_name}_page_{i+1}"}
        documents.append(Document(page_content=text, metadata=metadata))
    return documents

def load_pdfs_from_mongodb():
    documents = []
    target_filename = "Insure All The Way Website Information.pdf"

    grid_out = fs.find_one({"filename": target_filename})
    if not grid_out:
        raise FileNotFoundError(f"❌ '{target_filename}' not found in GridFS.")

    pdf_stream = BytesIO(grid_out.read())
    documents.extend(extract_text_from_pdf_stream(pdf_stream, target_filename))

    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def build_vector_db():
    docs = load_pdfs_from_mongodb()
    chunks = chunk_documents(docs)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local("faiss_index")
    return vectorstore

def load_vector_db():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

    if os.path.exists("faiss_index"):
        print("Loading existing FAISS vector DB...")
        vector_db = FAISS.load_local("faiss_index", embedding,  allow_dangerous_deserialization=True)
    else:
        print("No existing FAISS index found. Building new vector DB...")
        vector_db = build_vector_db()
        print("Vector DB built and saved.")

    return vector_db.as_retriever(search_kwargs={"k": 3})

retriever = load_vector_db()

def fetch_context(query: str, threshold: float = 0.3) -> str:
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return ""
    
    context_text = "\n\n".join([doc.page_content for doc in docs])
    return context_text


def ask_mistral(prompt: str, context: str = "", fallback_enabled=True) -> str:
    if context:
        full_prompt = (
            "You are an insurance assistant for Insure All The Way.\n"
            "Only answer using the information in the provided context.\n"
            "Do not add extra details, do not explain beyond what is provided.\n"
            "If the answer is not in the context, reply strictly with: "
            "'Insure All The Way does not currently offer this service.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{prompt}\n\n"
            "Answer in 1–3 sentences, direct and factual."
        )
    else:
        if fallback_enabled:
            full_prompt = (
                "You are an insurance assistant for Insure All The Way.\n"
                "If you cannot find relevant info in company documentation, reply strictly with: "
                "'Insure All The Way does not currently offer this service.'\n\n"
                f"Question:\n{prompt}\n\n"
                "Answer in 1–3 sentences, direct and factual."
            )
        else:
            return "<p>Sorry, I couldn't find any relevant information.</p>"


    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=500,
            temperature=0.3,
            top_p=0.9
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"<p><strong>Error:</strong> {e}</p>"


# === DB Utilities === #
def get_policy_by_registration(registration_number):
    policy = db["insurancePolicies"].find_one({"registrationNumber": registration_number})
    if not policy:
        return f"<p>No policy found for registration number: <strong>{registration_number}</strong></p>"

    user = db["users"].find_one({"_id": ObjectId(policy["user"])})
    agent = db["users"].find_one({"_id": ObjectId(policy.get("agent"))}) if "agent" in policy else None

    return (
        f"<h3>Policy for {registration_number}</h3><ul>"
        f"<li><strong>Type:</strong> {policy['insuranceType']}</li>"
        f"<li><strong>Duration:</strong> {policy['startDate']} – {policy['endDate']}</li>"
        f"<li><strong>User:</strong> {user.get('firstName')} {user.get('lastName')} ({user.get('email')})</li>"
        f"<li><strong>Agent:</strong> {agent.get('firstName', 'Not assigned')} ({agent.get('email', 'N/A')})</li>"
        "</ul>"
    )

def detect_intent(message: str, user_id=None):
    msg = message.lower()

    # Predefined intents
    if "registration" in msg or "plate number" in msg:
        words = message.split()
        reg_num = next((w for w in words if len(w) >= 6 and any(c.isdigit() for c in w)), None)
        return get_policy_by_registration(reg_num) if reg_num else "<p>Please provide a valid registration number.</p>"

    elif "email:" in msg:
        email = msg.split("email:")[1].strip()
        return get_policy_by_registration(email)

    elif any(kw in msg for kw in ["buy", "purchase", "get insurance", "price", "quote", "rate", "cost", "coverage", "enroll"]):
        return (
            "<p>You can explore our insurance offerings here:</p>"
            "<ul>"
            "<li><a href='https://insurealltheway.co/motor-insurance' target='_blank'>Motor Insurance</a></li>"
            "<li><a href='https://insurealltheway.co/health-insurance' target='_blank'>Health Insurance</a></li>"
            "<li><a href='https://insurealltheway.co/property-insurance' target='_blank'>Property Insurance</a></li>"
            "</ul>"
        )

    # === PRIMARY CONTEXTUAL ANSWER ===
    context = fetch_context(message)
    
    if context and len(context.strip()) > 50:
        response = ask_mistral(message, context)
        return f"<p>{response}</p>"
    
    fallback_response = ask_mistral(message, context="", fallback_enabled=True)
    return f"<p>{fallback_response}</p>"



# === Endpoint === #
@app.post("/chat")
def chat(req: ChatRequest):
    reply = detect_intent(req.message, req.user_id)
    return {"reply": reply, "format": "html"}

# === Optional: CLI to rebuild the DB === #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the vector DB")
    args = parser.parse_args()

    if args.rebuild:
        print("📦 Rebuilding vector DB from MongoDB GridFS PDF...")
        build_vector_db()
        print("✅ Vector DB rebuilt and saved to disk.")
