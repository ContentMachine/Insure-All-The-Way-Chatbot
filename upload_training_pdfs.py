# import os
# import tempfile
# import logging
# from pymongo import MongoClient
# from gridfs import GridFS
# from dotenv import load_dotenv
# from pathlib import Path

# # Set up logging
# logging.basicConfig(filename='upload_pdfs.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Check for langchain_community
# try:
#     from langchain_community.document_loaders import PyPDFLoader
#     LANGCHAIN_AVAILABLE = True
# except ImportError:
#     logging.error("langchain_community not installed. Install with: pip install langchain-community --user")
#     print("Warning: langchain_community not installed. Install with: pip install langchain-community --user")
#     LANGCHAIN_AVAILABLE = False

# # Check for pymongo
# try:
#     from pymongo import MongoClient
#     from gridfs import GridFS
#     PYMONGO_AVAILABLE = True
# except ImportError:
#     logging.error("pymongo not installed. Install with: pip install pymongo --user")
#     print("Warning: pymongo not installed. Install with: pip install pymongo --user")
#     PYMONGO_AVAILABLE = False

# # Load environment
# load_dotenv()
# MONGODB_URI = os.getenv("MONGODB_URI")
# MONGODB_DATABASE = os.getenv("MONGODB_DATABASE")
# if not (MONGODB_URI and MONGODB_DATABASE):
#     error_msg = "Missing MONGODB_URI or MONGODB_DATABASE in .env file."
#     logging.error(error_msg)
#     raise RuntimeError(error_msg)

# # Define PDFs to upload with metadata
# pdf_files = [
#     {
#         "path": "C:/Users/Professor Williams/Desktop/tobe_wdw/ChatBot2/trainingData/insurance-for-dummies.pdf",
#         "filename": "insurance-for-dummies.pdf",
#         "metadata": {
#             "insurance_type": "general",
#             "document_category": "training",
#             "source": "training_data",
#             "created_date": "2025-06-25"
#         }
#     },
#     {
#         "path": "C:/Users/Professor Williams/Desktop/tobe_wdw/ChatBot2/trainingData/Naicom Anti Money Laundering and Countering The Financing of Terrorism Regulations 2013.pdf",
#         "filename": "Naicom Anti Money Laundering and Countering The Financing of Terrorism Regulations 2013.pdf",
#         "metadata": {
#             "insurance_type": "general",
#             "document_category": "regulation",
#             "source": "training_data",
#             "created_date": "2025-06-25"
#         }
#     },
#     {
#         "path": "C:/Users/Professor Williams/Desktop/tobe_wdw/ChatBot2/trainingData/Motor Vehicle Third Party Insurance Act 1950.pdf",
#         "filename": "Motor Vehicle Third Party Insurance Act 1950.pdf",
#         "metadata": {
#             "insurance_type": "car",
#             "document_category": "regulation",
#             "source": "training_data",
#             "created_date": "2025-06-25"
#         }
#     },
#     {
#         "path": "C:/Users/Professor Williams/Desktop/tobe_wdw/ChatBot2/trainingData/MicroInsurance Guidelines 2018.pdf",
#         "filename": "MicroInsurance Guidelines 2018.pdf",
#         "metadata": {
#             "insurance_type": "general",
#             "document_category": "guideline",
#             "source": "training_data",
#             "created_date": "2025-06-25"
#         }
#     },
#     {
#         "path": "C:/Users/Professor Williams/Desktop/tobe_wdw/ChatBot2/trainingData/Insurance-Act-2003.pdf",
#         "filename": "Insurance-Act-2003.pdf",
#         "metadata": {
#             "insurance_type": "general",
#             "document_category": "regulation",
#             "source": "training_data",
#             "created_date": "2025-06-25"
#         }
#     },
#     {
#         "path": "C:/Users/Professor Williams/Desktop/tobe_wdw/ChatBot2/trainingData/GUIDELINES-FOR-MICRO-PENSION-PLAN.pdf",
#         "filename": "GUIDELINES-FOR-MICRO-PENSION-PLAN.pdf",
#         "metadata": {
#             "insurance_type": "pension",
#             "document_category": "guideline",
#             "source": "training_data",
#             "created_date": "2025-06-25"
#         }
#     },
#     {
#         "path": "C:/Users/Professor Williams/Desktop/tobe_wdw/ChatBot2/trainingData/Guideline for Insurance of Government Assets 2.pdf",
#         "filename": "Guideline for Insurance of Government Assets 2.pdf",
#         "metadata": {
#             "insurance_type": "general",
#             "document_category": "guideline",
#             "source": "training_data",
#             "created_date": "2025-06-25"
#         }
#     },
#     {
#         "path": "C:/Users/Professor Williams/Desktop/tobe_wdw/ChatBot2/trainingData/Approved_Corporate_Governance_Guideline_2021_y6vrpxZ.pdf",
#         "filename": "Approved_Corporate_Governance_Guideline_2021_y6vrpxZ.pdf",
#         "metadata": {
#             "insurance_type": "general",
#             "document_category": "guideline",
#             "source": "training_data",
#             "created_date": "2025-06-25"
#         }
#     },
#     {
#         "path": "C:/Users/Professor Williams/Desktop/tobe_wdw/ChatBot2/trainingData/CODE-OF-ETHICS-FOR-INSURANCE-PRACTITIONERS-IN-NIGERIA.pdf",
#         "filename": "CODE-OF-ETHICS-FOR-INSURANCE-PRACTITIONERS-IN-NIGERIA.pdf",
#         "metadata": {
#             "insurance_type": "general",
#             "document_category": "ethics",
#             "source": "training_data",
#             "created_date": "2025-06-25"
#         }
#     },
#     {
#         "path": "C:/Users/Professor Williams/Desktop/tobe_wdw/ChatBot2/trainingData/Pension Reform Act 2014.pdf",
#         "filename": "Pension Reform Act 2014.pdf",
#         "metadata": {
#             "insurance_type": "pension",
#             "document_category": "regulation",
#             "source": "training_data",
#             "created_date": "2025-06-25"
#         }
#     },
#     {
#         "path": "C:/Users/Professor Williams/Desktop/tobe_wdw/ChatBot2/trainingData/Revised-Guideline-on-GLIP-2020.pdf",
#         "filename": "Revised-Guideline-on-GLIP-2020.pdf",
#         "metadata": {
#             "insurance_type": "general",
#             "document_category": "guideline",
#             "source": "training_data",
#             "created_date": "2025-06-25"
#         }
#     },
#     {
#         "path": "C:/Users/Professor Williams/Desktop/tobe_wdw/ChatBot2/trainingData/Takaful Guidelines Approved.pdf",
#         "filename": "Takaful Guidelines Approved.pdf",
#         "metadata": {
#             "insurance_type": "takaful",
#             "document_category": "guideline",
#             "source": "training_data",
#             "created_date": "2025-06-25"
#         }
#     },
#     {
#         "path": "C:/Users/Professor Williams/Desktop/tobe_wdw/ChatBot2/trainingData/Revised-Regulation-on-RLA-2020.pdf",
#         "filename": "Revised-Regulation-on-RLA-2020.pdf",
#         "metadata": {
#             "insurance_type": "general",
#             "document_category": "regulation",
#             "source": "training_data",
#             "created_date": "2025-06-25"
#         }
#     },
#     {
#         "path": "C:/Users/Professor Williams/Desktop/tobe_wdw/ChatBot2/trainingData/MOTOR COMPREHENSIVE__2024__SPECIMEN 2.pdf",
#         "filename": "MOTOR COMPREHENSIVE__2024__SPECIMEN 2.pdf",
#         "metadata": {
#             "insurance_type": "car",
#             "document_category": "policy",
#             "source": "training_data",
#             "created_date": "2025-06-25"
#         }
#     },
#     {
#         "path": "C:/Users/Professor Williams/Desktop/tobe_wdw/ChatBot2/trainingData/Enhanced Third Party Insurance.pdf",
#         "filename": "Enhanced Third Party Insurance.pdf",
#         "metadata": {
#             "insurance_type": "car",
#             "document_category": "policy",
#             "source": "training_data",
#             "created_date": "2025-06-25"
#         }
#     },
#     {
#         "path": "C:/Users/Professor Williams/Desktop/tobe_wdw/ChatBot2/trainingData/Professional Indemnity.pdf",
#         "filename": "Professional Indemnity.pdf",
#         "metadata": {
#             "insurance_type": "indemnity",
#             "document_category": "policy",
#             "source": "training_data",
#             "created_date": "2025-06-25"
#         }
#     }
# ]

# # Upload and validate PDFs
# def upload_and_validate_pdf(pdf_info):
#     if not PYMONGO_AVAILABLE:
#         logging.error("pymongo not installed. Cannot upload PDFs.")
#         print("Error: pymongo not installed. Cannot upload PDFs.")
#         return False
#     if not LANGCHAIN_AVAILABLE:
#         logging.error("langchain_community not installed. Cannot validate PDFs.")
#         print("Error: langchain_community not installed. Cannot validate PDFs.")
#         return False

#     try:
#         # Connect to MongoDB
#         client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=30000)
#         db = client[MONGODB_DATABASE]
#         fs = GridFS(db)

#         # Check if PDF exists locally
#         pdf_path = Path(pdf_info["path"])
#         if not pdf_path.exists():
#             error_msg = f"PDF {pdf_path} not found."
#             logging.error(error_msg)
#             print(error_msg)
#             client.close()
#             return False

#         # Check if file already exists in GridFS to avoid duplicates
#         if fs.exists({"filename": pdf_info["filename"]}):
#             print(f"PDF {pdf_info['filename']} already exists in GridFS. Skipping upload.")
#             logging.info(f"PDF {pdf_info['filename']} already exists in GridFS.")
#         else:
#             # Upload PDF to GridFS
#             with open(pdf_path, "rb") as pdf_file:
#                 file_id = fs.put(
#                     pdf_file,
#                     filename=pdf_info["filename"],
#                     metadata=pdf_info["metadata"]
#                 )
#             print(f"Uploaded {pdf_info['filename']} to GridFS with ID: {file_id}")
#             logging.info(f"Uploaded {pdf_info['filename']} to GridFS with ID: {file_id}")

#         # Validate by retrieving and loading PDF
#         file = fs.find_one({"filename": pdf_info["filename"]})
#         if not file:
#             error_msg = f"PDF {pdf_info['filename']} not found in GridFS after upload."
#             logging.error(error_msg)
#             print(error_msg)
#             client.close()
#             return False

#         # Save to temporary file for PyPDFLoader
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#             temp_file.write(file.read())
#             temp_path = temp_file.name

#         try:
#             loader = PyPDFLoader(temp_path)
#             docs = loader.load()
#             print(f"Validated {pdf_info['filename']}: Loaded {len(docs)} pages.")
#             logging.info(f"Validated {pdf_info['filename']}: Loaded {len(docs)} pages.")
#         finally:
#             try:
#                 os.unlink(temp_path)
#             except Exception as e:
#                 warning_msg = f"Failed to delete temporary file {temp_path}: {e}"
#                 logging.warning(warning_msg)
#                 print(f"Warning: {warning_msg}")

#         client.close()
#         return True

#     except Exception as e:
#         error_msg = f"Error processing {pdf_info.get('filename', 'unknown')}: {e}"
#         logging.error(error_msg)
#         print(error_msg)
#         if 'client' in locals():
#             client.close()
#         return False

# # Main execution
# if __name__ == "__main__":
#     print("Uploading and validating PDFs to MongoDB GridFS...")
#     logging.info("Starting PDF upload and validation process.")
#     for pdf in pdf_files:
#         success = upload_and_validate_pdf(pdf)
#         if not success:
#             print(f"Failed to process {pdf.get('filename', 'unknown')}.")
#             logging.error(f"Failed to process {pdf.get('filename', 'unknown')}.")
#     print("Processing complete.")
#     logging.info("PDF upload and validation process completed.")



import os
from pymongo import MongoClient
from gridfs import GridFS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGODB_URI")
MONGO_DB = os.getenv("MONGODB_DATABASE")

# === Configuration === #
pdf_file_path = "/Users/mac/Desktop/Wdw/ChatBot2/trainingData/Insurance All The Way Training Data.pdf"  
file_name_in_mongo = "Insure All The Way Website Information.pdf" 

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
