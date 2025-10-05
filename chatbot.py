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
import fitz  
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from mistralai import Mistral
import re

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MONGO_URI = os.getenv("MONGODB_URI")
MONGO_DB = os.getenv("MONGODB_DATABASE")

mistral_client = Mistral(api_key=MISTRAL_API_KEY) 
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB]
fs = GridFS(db)

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
    target_filename = "Insure All The Way Website Information2.pdf"

    grid_out = fs.find_one({"filename": target_filename})
    if not grid_out:
        raise FileNotFoundError(f"'{target_filename}' not found in GridFS.")

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
        vector_db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
    else:
        print("No existing FAISS index found. Building new vector DB...")
        vector_db = build_vector_db()
        print("Vector DB built and saved.")

    return vector_db.as_retriever(search_kwargs={"k": 3})

retriever = load_vector_db()

def fetch_context(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return ""
    context_text = "\n\n".join([doc.page_content for doc in docs])
    return context_text


INTENTS = ["greeting", "farewell", "policy_lookup", "website_info", "general_qa"]

def classify_intent(message: str) -> str:
    try:
        prompt = f"""
You are Uju, an intent classifier for an insurance chatbot.

Classify the user message into ONE of these intents:
1. policy_lookup → if the user provides a registration number, policy ID, or email tied to their account.
2. website_info → if the user asks about products, services, or offerings of Insure All The Way 
   (e.g., "What policies do you offer?", "Do you provide third-party insurance?", "How can I renew?").
3. greeting → greetings like "hello", "hi", "hey".
4. farewell → goodbyes like "bye", "see you", "goodnight".
5. general_qa → anything else (casual questions, chit-chat, insurance in general).

User message: "{message}"

Respond with ONLY one of: policy_lookup, website_info, greeting, farewell, general_qa
"""
        response = mistral_client.chat.complete(
            model="mistral-small-2503",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        intent = response.choices[0].message.content.strip().lower()
        return intent if intent in INTENTS else "general_qa"
    except Exception as e:
        print("Intent classification failed:", e)
        return "general_qa"


def ask_mistral(prompt: str, context: str = "", fallback_enabled=True) -> str:
    if context:
        full_prompt = (
            "You are an insurance assistant for Insure All The Way.\n"
            "Answer ONLY using the information in the provided context.\n"
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
        response = mistral_client.chat.complete(
            model="mistral-small-2503",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.3,
            max_tokens=500,
            top_p=0.9
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"<p><strong>Error:</strong> {e}</p>"

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
    intent = classify_intent(message)

    if intent == "greeting":
        return "<p>Hello 👋! How can I help you with your insurance today?</p>"

    if intent == "farewell":
        return "<p>Goodbye! 👋 Stay safe and insured with Insure All The Way.</p>"

    if intent == "policy_lookup":
        email_match = re.search(r'[\w\.-]+@[\w\.-]+', message)
        if email_match:
            return get_policy_by_registration(email_match.group())
        reg_match = re.search(r'\b[A-Z0-9]{6,}\b', message.upper())
        if reg_match:
            return get_policy_by_registration(reg_match.group())
        return "<p>Please provide a valid registration number or email.</p>"

    if intent == "website_info":
        context = fetch_context(message)
        if context and len(context.strip()) > 50:
            response = ask_mistral(message, context)
            return f"<p>{response}</p>"
        else:
            return "<p>I couldn’t find that in our knowledge base, but you can explore our offerings here: <a href='https://insurealltheway.co/insurance-products' target='_blank'>Insurance Products</a></p>"

    if intent == "general_qa":
        response = ask_mistral(message, context='', fallback_enabled=True)
        return f"<p>{response}</p>"

    return "<p>Sorry, I didn’t quite understand that.</p>"


@app.post("/chat")
def chat(req: ChatRequest):
    reply = detect_intent(req.message, req.user_id)
    return {"reply": reply, "format": "html"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the vector DB")
    args = parser.parse_args()

    if args.rebuild:
        print("📦 Rebuilding vector DB from MongoDB GridFS PDF...")
        build_vector_db()
        print("✅ Vector DB rebuilt and saved to disk.")

