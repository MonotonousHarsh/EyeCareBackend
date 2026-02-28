import os
import shutil
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader

# --- CONFIGURATION ---
DB_NAME = "chroma_db"  # The single, unified database name
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- 1. CLEAN SLATE: REMOVE OLD DATABASES ---
print("🧹 Cleaning up old databases...")
for db in ["./medical_knowledge_db", "./medical_db", f"./{DB_NAME}"]:
    if os.path.exists(db):
        shutil.rmtree(db)
        print(f"   - Deleted {db}")

# --- 2. PREPARE EMBEDDINGS ---
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
all_docs = []

# --- 3. INGEST MASTER MARKDOWN (LOGIC & TRIAGE) ---
# This file is critical for definitions and decision trees
md_file = "medical_knowledge_base.md"
if os.path.exists(md_file):
    print(f"📄 Processing Master Logic: {md_file}...")
    with open(md_file, "r") as f:
        md_text = f.read()
    
    # Split by headers so the bot understands "Protocol A" vs "Protocol B"
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_chunks = markdown_splitter.split_text(md_text)
    all_docs.extend(md_chunks)
    print(f"   - Added {len(md_chunks)} logic chunks.")
else:
    print(f"⚠️ WARNING: {md_file} not found! Triage logic will be missing.")

# --- 4. INGEST RAW DOCUMENTS (DEEP DETAILS) ---
# These files provide the "textbook" knowledge for specific questions
files = [
    {"path": "Subconjunctival Hemorrhage Notes.docx", "loader": Docx2txtLoader},
    {"path": "Pterygium - notes (1).docx", "loader": Docx2txtLoader},
    {"path": "Stye Chalazion Assessment Guide.pdf", "loader": PyPDFLoader},
    {"path": "conjuctivites.docx", "loader": Docx2txtLoader}
]

raw_documents = []
for f in files:
    if os.path.exists(f["path"]):
        print(f"📖 Loading Document: {f['path']}...")
        loader_cls = f["loader"]
        # Handle PDF vs Docx loaders
        try:
            loader = loader_cls(f["path"])
            raw_documents.extend(loader.load())
        except Exception as e:
            print(f"   ❌ Failed to load {f['path']}: {e}")
    else:
        print(f"   ⚠️ File NOT FOUND: {f['path']} (Skipping)")

# Split raw documents into smaller searchable chunks
if raw_documents:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    doc_chunks = text_splitter.split_documents(raw_documents)
    all_docs.extend(doc_chunks)
    print(f"   - Added {len(doc_chunks)} detail chunks.")

# --- 5. BUILD THE DATABASE ---
print(f"💾 Saving {len(all_docs)} total chunks to {DB_NAME}...")
if all_docs:
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=f"./{DB_NAME}",
        collection_name="medical_docs"
    )
    print("✅ SUCCESS: Database rebuilt from zero!")
else:
    print("❌ ERROR: No documents were found. Database not created.")