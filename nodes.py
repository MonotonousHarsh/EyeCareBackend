import os
import json
from dotenv import load_dotenv

# --- LANGCHAIN & AI IMPORTS ---
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# FIXED IMPORT: Added AIMessage here
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# --- PROJECT IMPORTS ---
from state import AgentState
from protocol_nodes import PROTOCOLS
from db_mock import fetch_doctors

# 1. Load Env
load_dotenv()

# 2. Setup LLM (Llama 3.3 for speed & accuracy)
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0, 
    api_key=os.getenv("GROQ_API_KEY")
)

# 3. Setup RAG Database Connection
DB_FOLDER = "./chroma_db" 

# Initialize Embeddings
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to ChromaDB
vectorstore = Chroma(
    persist_directory=DB_FOLDER,
    embedding_function=embedding_function,
    collection_name="medical_docs"
)

# Create Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) 

# ==========================================
# NODE 1: TRIAGE (INTELLIGENT SUPERVISOR)
# ==========================================
def triage_node(state: AgentState):
    """
    INTELLIGENT ROUTER (Supervisor).
    Decides the next node based on Conversation + Video State + Active Protocols.
    """
    try:
        messages = state["messages"]
        user_msg = messages[-1].content.lower()
        
        # 1. SHORT-CIRCUIT: PROTOCOLS
        active_protocol = state.get("active_protocol")
        if active_protocol:
            print(f"🧠 SUPERVISOR: Active Protocol '{active_protocol}' detected. Continuing...")
            return {"intent": "assessment_protocol"}


            # 2. SHORT-CIRCUIT: DIAGNOSTIC DATA RECEIVED
        if state.get("functional_test_results"):
            print("🧠 SUPERVISOR: Received Vision SDK data. Routing directly to analysis...")
            return {"intent": "functional_vision_analysis"}
        # ----------------------------------

        # 2. DEFINE AVAILABLE TOOLS (UPDATED)
        tools = """
        1. "structural_vision_analysis": Use this if the user describes surface-level symptoms (redness, bumps, itching, "pink eye", "stye") OR asks you to check a visual anomaly on the outside of their eye.
        2. "functional_vision_analysis": PRIORITY tool if the user describes neurological or movement symptoms: dizziness, double vision, hitting their head, trauma, or delayed pupil reactions.
        3. "medical_advice": Use this for general theoretical questions ("What is Glaucoma?") where no visual test is needed.
        4. "find_doctor": Use this if user explicitly wants to find a specialist.
        5. "booking": Use this if user wants to book an appointment.
        6. "general_chat": For greetings or non-medical topics.
        """

        # 3. SUPERVISOR PROMPT (UPDATED)
        system_prompt = f"""
        You are the Supervisor AI for a state-of-resolutions Real-Time Eye Care Agent.
        
        AVAILABLE TOOLS:
        {tools}
        
        YOUR JOB:
        Analyze the user's last message to decide which medical instrument or tool to activate next.
        
        CRITICAL ROUTING RULES:
        1. **SURFACE VS NEUROLOGICAL**: If symptoms are physical/surface (redness, bumps, swelling), choose "structural_vision_analysis". If symptoms are neurological/behavioral (dizziness, double vision, head trauma), choose "functional_vision_analysis".
        2. **NO MORE REQUEST_IMAGE**: Do not ask the user to upload an image. The Vision Agent SDK handles the camera feed automatically based on your tool choice.
        3. Only choose "medical_advice" if the user asks a theoretical question unrelated to a current physical symptom.
        
        OUTPUT JSON ONLY:
        {{
            "reasoning": "User mentioned feeling dizzy after a fall -> Neurological test needed",
            "next_step": "functional_vision_analysis" 
        }}
        """

        # 4. INVOKE LLM
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_msg)
        ])
        
        # 5. PARSE JSON
        content = response.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")
        elif content.startswith("```"):
            content = content.replace("```", "")
        
        try:
            decision = json.loads(content)
        except json.JSONDecodeError:
            print(f"❌ JSON Parse Error. Raw content: {content}")
            return {"intent": "general_chat"}

        next_step = decision.get("next_step", "general_chat")
        print(f"🧠 INTELLIGENT ROUTER DECISION: {decision}")
        
        # Pass the intended test type if it's a functional analysis
        functional_test_type = None
        if next_step == "functional_vision_analysis":
             if "dizzy" in user_msg or "double" in user_msg:
                 functional_test_type = "nystagmus_test"
             else:
                 functional_test_type = "plr_test" # Default to Pupillary Light Reflex for trauma

        return {
            "intent": next_step,
            "functional_test_type": functional_test_type
        }

    except Exception as e:
        print(f"❌ Triage Error: {e}")
        return {"intent": "general_chat"}
# ==========================================
# NODE 2: MEDICAL ADVICE (The RAG System)
# ==========================================
def medical_advice_node(state: AgentState):
    """
    Generalized RAG Node.
    """
    try:
        messages = state["messages"]
        original_query = messages[-1].content
        
        # --- STEP 1: GENERALIZED QUERY REWRITING ---
        if len(messages) > 2:
            print("🤔 Context Analysis: Rewriting query based on history...")
            
            rephrase_prompt = ChatPromptTemplate.from_messages([
                ("system", "Given a chat history and the latest user question, formulate a standalone question."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ])
            
            history_window = messages[-6:-1] 
            chain = rephrase_prompt | llm | StrOutputParser()
            search_query = chain.invoke({"chat_history": history_window, "question": original_query})
            print(f"🔄 Transformation: '{original_query}' -> '{search_query}'")
        else:
            search_query = original_query

        # --- STEP 2: SEARCH ---
        print(f"🔍 Searching DB for: {search_query}")
        docs = retriever.invoke(search_query)
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        if not context_text:
            return {"messages": [AIMessage(content="I checked my medical database, but I couldn't find specific details on that.")]}

        # --- STEP 3: ANSWER ---
        answer_prompt = ChatPromptTemplate.from_template("""
        You are an expert Ophthalmologist AI. Use the medical context below to answer.
        
        CONTEXT:
        {context}
        
        USER QUESTION: 
        {question}
        
        ANSWER:
        """)
        
        chain = answer_prompt | llm | StrOutputParser()
        response = chain.invoke({"context": context_text, "question": original_query})
        
        # *** THE FIX IS HERE ***
        # Changed HumanMessage -> AIMessage
        return {"messages": [AIMessage(content=response)]}

    except Exception as e:
        print(f"❌ RAG Error: {e}")
        return {"messages": [AIMessage(content="I'm having trouble accessing my memory right now.")]}

# ==========================================
# NODE 3: DOCTOR FINDER
# ==========================================
def find_doctor_node(state: AgentState):
    user_profile = state.get("user_profile")
    city = user_profile.get("city", "Agra") if user_profile else "Agra"
    condition = state.get("active_protocol") or state.get("detected_condition") or "General Eye Care"
    
    # Map technical protocol names
    if condition == "sch_protocol": condition = "Subconjunctival Hemorrhage"
    elif condition == "stye_protocol": condition = "Stye/Chalazion"
    elif condition == "conjunctivitis_protocol": condition = "Conjunctivitis"
    elif condition == "pterygium_protocol": condition = "Pterygium"
    
    print(f"🔎 Searching doctors in {city} for {condition}...")
    doctors = fetch_doctors(city, condition)
    
    if not doctors:
        return {"messages": [AIMessage(content=f"I couldn't find any specialists in {city}.")]}
    
    response_lines = [f"Here are the top specialists in {city} for you:"]
    for doc in doctors:
        specialties = ", ".join(doc['specialties'])
        response_lines.append(f"👨‍⚕️ **{doc['name']}** ({doc['clinic']})\n   - Specialty: {specialties}\n   - Rating: ⭐ {doc['rating']}\n   - Next Slot: {doc['next_available']}")
    
    response_lines.append("\nType 'Book [Doctor Name]' to schedule an appointment.")
    return {"messages": [AIMessage(content="\n\n".join(response_lines))]}

# ==========================================
# NODE 4: BOOKING AGENT
# ==========================================
def booking_node(state: AgentState):
    msg = state["messages"][-1].content
    selected_doc = "Dr. Specialist"
    if "Sharma" in msg: selected_doc = "Dr. Aditi Sharma"
    elif "Verma" in msg: selected_doc = "Dr. Rajesh Verma"
    return {"messages": [AIMessage(content=f"✅ **Booking Confirmed!**\n\nAppointment with {selected_doc}\nDate: Tomorrow\nTime: 10:00 AM")]}

# ==========================================
# NODE 5: GENERAL CHAT
# ==========================================
def general_chat_node(state: AgentState):
    return {"messages": [AIMessage(content="I am the EyeCare AI. How can I assist you today?")]}