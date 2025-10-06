import os
from pymongo import MongoClient
from gridfs import GridFS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGODB_URI")
MONGO_DB = os.getenv("MONGODB_DATABASE")

# === Configuration === #
pdf_file_path = "/Users/mac/Desktop/Wdw/ChatBot2/trainingData/INSURE ALL THE WAY TRAINING DATA.pdf"  
file_name_in_mongo = "Insure All The Way Website Information2.pdf" 

# === Connect to MongoDB === #
client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
fs = GridFS(db)

# === Upload PDF to GridFS === #
with open(pdf_file_path, "rb") as f:
    existing = fs.find_one({"filename": file_name_in_mongo})
    if existing:
        fs.delete(existing._id)
        print(f"Deleted old version of '{file_name_in_mongo}' from GridFS.")

    file_id = fs.put(f, filename=file_name_in_mongo)
    print(f"✅ Uploaded '{file_name_in_mongo}' to MongoDB GridFS with ID: {file_id}")
