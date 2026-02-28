import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# 1. Load the API Key from the .env file
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env file")

# 2. Initialize Groq
# We use 'llama3-70b-8192' which is faster and smarter than the local version
llm = ChatGroq(
    model="llama3-70b-8192", 
    temperature=0,
    api_key=api_key
)

print("✅ Groq (Llama 3) Model Loaded Successfully")