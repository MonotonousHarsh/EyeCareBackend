import os
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Define KNOWLEDGE FILES ONLY
files = [
    {"path": "Subconjunctival Hemorrhage Notes.docx", "loader": Docx2txtLoader},
    {"path": "Pterygium - notes (1).docx", "loader": Docx2txtLoader},
    {"path": "Stye Chalazion Assessment Guide.pdf", "loader": PyPDFLoader},
    {"path": "conjuctivites.docx", "loader": Docx2txtLoader} # NEWLY ADDED
]

documents = []
for f in files:
    if os.path.exists(f["path"]):
        print(f"📖 Loading {f['path']}...")
        loader = f["loader"](f["path"])
        documents.extend(loader.load())
    else:
        print(f"⚠️ File NOT FOUND: {f['path']} (Skipping)")

# 2. Split & Embed
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Clear old DB
if os.path.exists("./medical_db"):
    import shutil
    shutil.rmtree("./medical_db")

db = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory="./medical_db")
print(f"✅ Knowledge Base Rebuilt! {len(chunks)} chunks ingested.")