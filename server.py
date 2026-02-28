import os
import shutil
import uuid
import asyncio
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

# --- PROJECT IMPORTS ---
from main import graph 
from langchain_core.messages import HumanMessage, AIMessage

# Make sure you have created vision_worker.py in the same directory!
from vision_worker import run_diagnostic_agent 

# --- STREAM SDK IMPORT ---
# pip install getstream
from getstream import Stream

# --- CONFIGURATION ---
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EyeCareAPI")

app = FastAPI(title="Agentic EyeCare API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SECURE CREDENTIALS ---
STREAM_API_KEY = os.getenv("STREAM_API_KEY", "YOUR_STREAM_API_KEY")
STREAM_API_SECRET = os.getenv("STREAM_API_SECRET", "YOUR_STREAM_API_SECRET")
stream_client = Stream(api_key=STREAM_API_KEY, api_secret=STREAM_API_SECRET)

# --- IN-MEMORY DB FOR WEBHOOKS ---
DIAGNOSTIC_RESULTS_DB: Dict[str, Any] = {}

# ==========================================
# REQUEST & RESPONSE MODELS
# ==========================================
class ChatRequest(BaseModel):
    user_id: str
    thread_id: str
    message: Optional[str] = None
    image_id: Optional[str] = None 
    functional_test_results: Optional[Dict[str, Any]] = None
    functional_test_type: Optional[str] = None 

class ChatResponse(BaseModel):
    response: str
    active_protocol: Optional[str] = None
    triage_status: Optional[str] = None
    video_stream_active: Optional[bool] = False

class AgentWebhookPayload(BaseModel):
    call_id: str
    test_type: str
    results: Dict[str, Any]

# ==========================================
# 1. LANGGRAPH CHAT ENDPOINT
# ==========================================
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        inputs = {}
        
        if request.message:
            inputs["messages"] = [HumanMessage(content=request.message)]
        
        if request.image_id:
            inputs["uploaded_image_id"] = request.image_id
            inputs["ai_prediction"] = None   
            inputs["active_protocol"] = None 
            inputs["protocol_step"] = None   
            if not request.message:
                inputs["messages"] = [HumanMessage(content="[User uploaded a static image]")]

        if request.functional_test_results:
            inputs["functional_test_results"] = request.functional_test_results
            if not request.message:
                test_name = request.functional_test_type or "video"
                inputs["messages"] = [HumanMessage(content=f"[Vision SDK completed {test_name} scan and returned data]")]
        
        if request.functional_test_type:
            inputs["functional_test_type"] = request.functional_test_type

        config = {"configurable": {"thread_id": request.thread_id}}
        output = graph.invoke(inputs, config=config)
        
        # Robust Response Extraction
        all_messages = output["messages"]
        bot_responses = []
        last_human_idx = -1
        for i in range(len(all_messages) - 1, -1, -1):
            if isinstance(all_messages[i], HumanMessage):
                last_human_idx = i
                break
        
        if last_human_idx != -1:
            for msg in all_messages[last_human_idx+1:]:
                if isinstance(msg, AIMessage) and msg.content.strip():
                    bot_responses.append(msg.content)
        
        if not bot_responses and len(all_messages) > 0:
            last_msg = all_messages[-1]
            if isinstance(last_msg, AIMessage) and last_msg.content.strip():
                bot_responses.append(last_msg.content)

        full_response = "\n\n".join(bot_responses) or "I processed your data. Please tell me if you have any other symptoms."

        return {
            "response": full_response,
            "active_protocol": output.get("active_protocol"),
            "triage_status": output.get("intent"),
            "video_stream_active": output.get("video_stream_active", False)
        }
    except Exception as e:
        logger.error(f"Graph execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 2. STREAM VIDEO TOKEN GENERATOR
# ==========================================
def launch_agent_task(call_id: str, test_type: str):
    """Wrapper to run the async agent in a background thread."""
    asyncio.run(run_diagnostic_agent(call_id, test_type))

@app.get("/generate-video-token")
async def generate_video_token(
    background_tasks: BackgroundTasks, 
    user_id: str = Query(..., description="The unique ID of the patient")
):
    try:
        token = stream_client.create_token(user_id, expiration=3600)
        call_id = f"diagnostic_scan_{uuid.uuid4().hex[:8]}"
        
        # Fire and forget: Launch the Python Vision Agent instantly!
        background_tasks.add_task(launch_agent_task, call_id, "plr_test")
        
        return {
            "token": token,
            "call_id": call_id,
            "user_id": user_id,
            "api_key": STREAM_API_KEY
        }
    except Exception as e:
        logger.error(f"Token generation failed: {e}")
        raise HTTPException(status_code=500, detail="Could not generate secure video token")

# ==========================================
# 3. PYTHON AGENT WEBHOOK & REACT POLLING
# ==========================================
@app.post("/agent-webhook")
async def receive_agent_data(payload: AgentWebhookPayload):
    print(f"📥 Received ultra-low latency data for Room: {payload.call_id}")
    DIAGNOSTIC_RESULTS_DB[payload.call_id] = payload.results
    return {"status": "success", "message": "Telemetry received."}

@app.get("/diagnostic-results/{call_id}")
async def check_diagnostic_results(call_id: str):
    if call_id in DIAGNOSTIC_RESULTS_DB:
        results = DIAGNOSTIC_RESULTS_DB.pop(call_id)
        return {"status": "complete", "data": results}
    return {"status": "pending"}

# ==========================================
# 4. STATIC IMAGE UPLOAD (Unchanged)
# ==========================================
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        file_extension = file.filename.split(".")[-1]
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"image_id": file_path, "message": "Image uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Image upload failed")

@app.get("/")
def health_check():
    return {"status": "running", "service": "Agentic EyeCare Backend v3.0"}