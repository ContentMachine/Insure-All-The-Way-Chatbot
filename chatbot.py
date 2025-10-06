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
from bson import ObjectId
from datetime import datetime
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


INTENTS = ["greeting", "farewell", "policy_lookup", "website_info", "general_qa", "user_policy_summary"]


def classify_intent(message: str) -> str:
    try:
        prompt = f"""
You are Uju, an intent classifier for an insurance chatbot.

Classify the user's message into ONE of these intents:
1. user_policy_summary → if the user asks to view, check, or list their policies (e.g. "Show my policies", "What policies do I have?").
2. policy_lookup → if the user provides a registration number, policy ID, or email tied to their account.
3. website_info → if the user asks about products, services, or offerings of Insure All The Way.
4. greeting → greetings like "hello", "hi", "hey".
5. farewell → goodbyes like "bye", "see you".
6. general_qa → anything else.

User message: "{message}"

Respond with ONLY one of: user_policy_summary, policy_lookup, website_info, greeting, farewell, general_qa
"""
        response = mistral_client.chat.complete(
            model="mistral-small-2503",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
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
    "You are Uju, the official insurance assistant for Insure All The Way.\n"
    "Your main rule is to ONLY answer using the information from the provided context.\n"
    "If the answer cannot be found in the context, reply exactly with:\n"
    "'I could not find that information in Insure All The Way’s records.'\n\n"
    "Be careful, accurate, and truthful.\n"
    "Think step by step before answering — review all context and extract the most relevant details.\n"
    "Do not guess, assume, or add information that is not explicitly stated.\n"
    "Never contradict the provided context.\n\n"
    f"Context:\n{context}\n\n"
    f"User Question:\n{prompt}\n\n"
    "Now, based only on the context, provide a clear, professional, and factual answer in 2–4 sentences."
)
    else:
        if fallback_enabled:
         full_prompt = (
    "You are Uju, the official insurance assistant for Insure All The Way.\n"
    "You are answering a general question about the company’s services.\n"
    "If you cannot find relevant information in company documentation, reply exactly with:\n"
    "'I could not find that information in Insure All The Way’s records.'\n\n"
    "Be calm and professional. Do not make assumptions or guesses.\n"
    "Focus only on verified information about Insure All The Way.\n\n"
    f"User Question:\n{prompt}\n\n"
    "Respond in a short, clear, and factual manner (2–4 sentences)."
)
        else:
            return "<p>Sorry, I couldn't find any relevant information.</p>"

    try:
        response = mistral_client.chat.complete(
            model="mistral-small-2503",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.1,
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


def get_user(user_identifier):
    """Fetch a user by email or user_id."""
    if not user_identifier:
        return None
    try:
        if ObjectId.is_valid(user_identifier):
            return db["users"].find_one({"_id": ObjectId(user_identifier)})
        else:
            return db["users"].find_one({"email": user_identifier.lower()})
    except Exception as e:
        print("Error fetching user:", e)
        return None 
    


def get_policies_by_user(user_id):
    """Fetch all policies belonging to a user, formatted for display."""
    try:
        cursor = db["insurancePolicies"].find({"user": ObjectId(user_id)})
        policies = list(cursor)

        if not policies:
            return "<p>No insurance policies found for this user.</p>"

        policy_list = ""
        for policy in policies:
            # Fetch agent
            agent = db["users"].find_one({"_id": ObjectId(policy["agent"])}) if policy.get("agent") else None

            # Format insurance type (de-hyphenate and title-case)
            insurance_type = policy.get("insuranceType", "N/A").replace("-", " ").title()


            # Certificate link (only if available)
            certificate = policy.get("certificate")
            certificate_html = (
                f"<a href='{certificate}' target='_blank' style='color:#007BFF;text-decoration:none;'>Click here</a>"
                if certificate else "Not available"
            )

            # Created date
            created_at = policy.get("createdAt")
            created_at_str = created_at.strftime('%Y-%m-%d') if isinstance(created_at, datetime) else str(created_at or "N/A")

            # Agent name
            agent_name = (
                f"{agent.get('firstName', '')} {agent.get('lastName', '')}".strip()
                if agent else "Not assigned"
            )

            # Build HTML
            policy_list += (
                f"<div style='background:#f9f9f9;border:1px solid #ddd;border-radius:10px;"
                f"padding:15px;margin-block:15px;font-family:General Sans, sans-serif;'>"
                f"<h4 style='margin-bottom:10px;color:#333;'>Policy Number: {policy.get('policyNumber', 'N/A')}</h4>"
                f"<ul style='list-style:none;padding-left:0;line-height:1.6;margin:0;'>"
                f"<li><strong>Type:</strong> {insurance_type}</li>"
                f"<li><strong>Agent:</strong> {agent_name}</li>"
                f"<li><strong>Certificate:</strong> {certificate_html}</li>"
                f"<li><strong>Created At:</strong> {created_at_str}</li>"
                f"</ul></div>"
            )

        # Generate summary (optional AI step)
        summary = generate_policy_summary_with_mistral(policy_list)

        return (
            "<h3 style='font-family:General Sans, sans-serif;'>Your Policy Summary</h3>"
            f"{summary}"
            "<hr style='margin:15px 0;'>"
            "<p style='font-family:General Sans, sans-serif;margin-bottom:15px '>Here are all the policies linked to your account:</p>"
            f"{policy_list}"
        )

    except Exception as e:
        return f"<p>Error fetching policies: {e}</p>"



def detect_intent(message: str, user_id=None):
    intent = classify_intent(message)

    if intent == "greeting":
        return "<p>Hello 👋! How can I help you with your insurance today?</p>"
    
    if intent == "user_policy_summary":
        if not user_id:
            return "<p>Please log in to view your policy summary.</p>"

        user = get_user(user_id)
        if not user:
            return "<p>We couldn’t find your user profile. Please try logging in again.</p>"

        return get_policies_by_user(user["_id"])

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

def generate_policy_summary_with_mistral(formatted_policies: str) -> str:
    """Generate a warm, personalized summary of the user's policies using Mistral Small."""
    if not formatted_policies or "No insurance policies found" in formatted_policies:
        return "<p>You don’t currently have any active policies with Insure All The Way.</p>"

    summary_prompt = f"""
You are Uju, the friendly insurance assistant for Insure All The Way.

The following HTML represents a list of the user's insurance policies. 
Your job is to analyze this data and write a **personalized summary** (in one or two short paragraphs).

Be sure to:
- Mention how many policies the user has and what types they are.
- Highlight any unpaid or missing-certificate policies.
- If some policies look old or might need renewal soon, gently mention that.
- Use a warm, reassuring tone — sound like a helpful human assistant, not a robot.
- Keep it under 6 sentences.
- End with a positive suggestion or encouragement to review or renew.

Do not invent data not present in the input. Rely strictly on the information given.

User Policy Data:
{formatted_policies}
"""

    try:
        response = mistral_client.chat.complete(
            model="mistral-small-2503",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.5,
            max_tokens=350,
            top_p=0.9
        )
        return f"<p>{response.choices[0].message.content.strip()}</p>"
    except Exception as e:
        return f"<p><strong>Error generating summary:</strong> {e}</p>"



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

