import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Step 1: Load environment
load_dotenv()
API_KEY = os.getenv("HF_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing Hugging Face API key. Set HF_API_KEY in your .env file.")

# Step 2: Load PDF document
PDF_PATH = Path("C:/Users/Professor Williams/Desktop/tobe_wdw/ChatBot2/trainingData/insurance-for-dummies.pdf")
print("Loading document...")
try:
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
except Exception as e:
    raise RuntimeError(f"Failed to load PDF: {e}")

# Step 3: Chunk the document
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Step 4: Embed and store in vector DB
print("Embedding and storing chunks in Chroma...")
try:
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    raise RuntimeError(f"Failed to load embedding model: {e}")
if os.path.exists("./db"):
    shutil.rmtree("./db")  # Clear existing DB
db = Chroma.from_documents(chunks, embedding, persist_directory="./db")

# Step 5: Setup Mistral client
client = InferenceClient(api_key=API_KEY)

def build_prompt(user_question, context):
    return f"""[INST] You are a helpful assistant. Answer the question based only on the context below.

Context:
{context}

Question: {user_question} [/INST]"""

def ask_question(question):
    print("Searching for relevant context...")
    docs = db.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    print("Retrieved context:\n", context, "\n")

    prompt = build_prompt(question, context)

    print("Asking the Chatbot...\n")
    try:
        result = client.text_generation(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            prompt=prompt,
            max_new_tokens=200,
            temperature=0.7
        )
        if isinstance(result, str):
            print("Mistral:", result.strip(), "\n")
        else:
            print("Mistral:", getattr(result, "generated_text", str(result)).strip(), "\n")
    except Exception as e:
        print(f"Error querying model: {e}\n")

if __name__ == "__main__":
    print("Chatbot is ready to answer questions from your PDF using Mistral. Ask me anything (type 'exit' or 'quit' to quit):\n")
    while True:
        q = input("Your question: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        ask_question(q)