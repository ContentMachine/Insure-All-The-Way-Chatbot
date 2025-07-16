import os
import logging
from dotenv import load_dotenv
from pymongo import MongoClient
from gridfs import GridFS
from io import BytesIO
# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
import fitz
from langchain.schema import Document
from bson import ObjectId

# ---- SETUP LOGGING ---- #
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---- LOAD ENV VARIABLES ---- #
load_dotenv()
HF_TOKEN = os.getenv("HF_API_KEY")
MONGO_URI = os.getenv("MONGODB_URI")
MONGO_DB = os.getenv("MONGODB_DATABASE")
CHROMA_DIR = "chroma_db"

if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN not found in .env file")

# ---- SETUP HUGGING FACE INFERENCE CLIENT ---- #
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token=HF_TOKEN
)

# ---- CONNECT TO MONGODB ---- #
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB]
fs = GridFS(db)

# ---- FUNCTIONS ---- #


def extract_text_from_pdf_stream(pdf_stream, source_name):
    """Reads PDF bytes stream and returns LangChain documents."""
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
    for grid_out in fs.find():
        filename = grid_out.filename
        logging.info(f"Loading {filename} from MongoDB")
        pdf_stream = BytesIO(grid_out.read())
        docs = extract_text_from_pdf_stream(pdf_stream, filename)
        documents.extend(docs)

    return documents

def get_policy_by_registration(registration_number):
    policy = db["insurancePolicies"].find_one({"registrationNumber": registration_number})
    if not policy:
        return f"No policy found for registration number {registration_number}."

    # Get user info
    user = db["users"].find_one({"_id": ObjectId(policy["user"])})
    agent = db["users"].find_one({"_id": ObjectId(policy["agent"])}) if "agent" in policy else None

    return (
        f"📄 Policy Information for Registration Number: {registration_number}\n"
        f"- Policy Type: {policy['insuranceType']}\n"
        f"- Start Date: {policy['startDate']}\n"
        f"- End Date: {policy['endDate']}\n"
        f"- User: {user.get('firstName', 'Unknown')} {user.get('lastName', 'Unknown')} ({user.get('email', 'No email')})\n"
        f"- Agent: {agent.get('firstName', 'Not assigned')} {agent.get('lastName', 'Not assigned')} ({agent.get('email') if agent else 'N/A'})"
    )

def get_policies_by_user(user_id):
    policies = list(db["insurancePolicies"].find({"user": user_id}))
    if not policies:
        return f"No policies found for user {user_id}."
    
    response = f"Policies for user {user_id}:\n"
    for p in policies:
        response += f"- {p['registrationNumber']}: {p['insuranceType']} ({p['startDate']} to {p['endDate']})\n"
    return response.strip()

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    logging.info(f"Split documents into {len(chunks)} chunks.")
    return chunks

def build_vector_db(documents):
    chunks = chunk_documents(documents)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    db = Chroma.from_documents(chunks, embedding, persist_directory=CHROMA_DIR)
    logging.info(f"Stored {len(chunks)} chunks in Chroma vector store.")
    return db

def ask_mistral(prompt):
    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7,
            top_p=0.95
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        logging.error(f"Error querying model: {e}")
        return "\U0001F916 Assistant: Error contacting inference API."

def main():
    docs = load_pdfs_from_mongodb()
    db_vectordb = build_vector_db(docs)

    while True:
        query = input("Ask me an insurance question (type 'exit' to quit):\nYou: ")
        if query.lower() == "exit":
            break

        # Basic intent checks
        if "registration" in query.lower():
            # Extract registration number
            words = query.split()
            reg_num = next((w for w in words if len(w) >= 6 and any(c.isdigit() for c in w)), None)
            if reg_num:
                print("\U0001F916 Assistant:", get_policy_by_registration(reg_num))
                continue
            else:
                print("\U0001F916 Assistant: Please provide a valid registration number.")
                continue

        elif "my policies" in query.lower() or "user" in query.lower():
            # Dummy user ID for now (you can pass this from frontend later)
            user_id = "12345"
            print("\U0001F916 Assistant:", get_policies_by_user(user_id))
            continue

        # Fallback to LLM
        print("\U0001F916 Assistant:", ask_mistral(query))

if __name__ == "__main__":
    main()
