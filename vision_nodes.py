import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import os
import json
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from state import AgentState
from dotenv import load_dotenv

load_dotenv()

# --- 1. SETUP LLM ---
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0.6, 
    api_key=os.getenv("GROQ_API_KEY")
)

# --- 2. MODEL SETUP (PyTorch - Structural Analysis) ---
CLASSES = ['healthy', 'pterygium', 'conjunctivitis', 'pinguecula'] 
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_vit():
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    model.head = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.head.in_features, NUM_CLASSES)
    )
    return model

# Load Weights
MODEL_PATH = "best_vit_base_4class.pth" 
model = create_vit().to(DEVICE)

if os.path.exists(MODEL_PATH):
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"✅ Structural Vision Model Loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Model Load Error: {e}")
else:
    print(f"⚠️ Warning: Model file not found at {MODEL_PATH}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# NODE A: STRUCTURAL VISION (The Old Reliable)
# ==========================================
def structural_vision_analysis_node(state: AgentState):
    """
    Analyzes static image frames for surface diseases using PyTorch.
    """
    image_path = state.get("uploaded_image_id")
    messages = state["messages"]
    user_question = messages[-1].content 
    
    if not image_path or not os.path.exists(image_path):
        return {"messages": [AIMessage(content="The camera feed didn't capture a clear frame. Please align your eye and let the agent scan again.")]}

    print(f"👁️ Structural AI Processing Frame: {image_path}")

    # A. RUN PREDICTION
    try:
        img = Image.open(image_path).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(img_t)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        detected_class = CLASSES[predicted_idx.item()]
        conf_score = confidence.item()
        
    except Exception as e:
        print(f"❌ Vision Error: {e}")
        return {"messages": [AIMessage(content="I encountered a technical error analyzing the surface of your eye.")]}

    # B. MAP TO PROTOCOL
    next_protocol = None
    if detected_class == "conjunctivitis": next_protocol = "conjunctivitis_protocol"
    elif detected_class == "pterygium": next_protocol = "pterygium_protocol"
    elif detected_class == "pinguecula": next_protocol = "pinguecula_protocol" 
    
    if detected_class == "healthy":
        return {
            "ai_prediction": {"class": "healthy", "confidence": conf_score},
            "active_protocol": None,
            "messages": [AIMessage(content="✅ **Surface Scan Complete:** The visible surface of your eye appears healthy! I didn't detect signs of conjunctivitis or growths.")]
        }

    # C. GENERATE DOCTOR-LIKE EXPLANATION
    system_prompt = f"""
    You are an expert AI Ophthalmologist. You just analyzed a patient's live camera frame.
    FINDING: {detected_class.upper()} (Confidence: {conf_score:.1%})
    
    YOUR GOAL: Tell the user what you found on the surface of their eye, answer their specific question ("{user_question}"), and explain the visual evidence. End by stating you need to ask a few questions to confirm.
    Keep it professional, empathetic, and under 150 words.
    """
    
    explanation = llm.invoke([SystemMessage(content=system_prompt)])
    
    return {
        "ai_prediction": {"class": detected_class, "confidence": conf_score},
        "active_protocol": next_protocol,
        "protocol_step": "start",
        "messages": [AIMessage(content=explanation.content)]
    }

# ==========================================
# NODE B: FUNCTIONAL VISION (The Hackathon Star)
# ==========================================
def functional_vision_analysis_node(state: AgentState):
    """
    Coordinates with the Vision Agents SDK for real-time video diagnostics (PLR, Nystagmus).
    """
    test_type = state.get("functional_test_type", "plr_test")
    test_results = state.get("functional_test_results")
    
    # 1. TRIGGER THE SDK: If we don't have results yet, tell the frontend to start the test!
    if not test_results:
        print(f"🎥 Initiating Live Video Test: {test_type}")
        
        if test_type == "plr_test":
            instruction = "I need to check your pupillary reflexes. Please hold your camera steady in front of your eye. I am going to briefly flash the screen to measure how fast your pupil reacts."
        else:
            instruction = "I need to check your eye movement. Please keep your head completely still and follow the moving dot on your screen with your eyes."
            
        return {
            "video_stream_active": True,
            "messages": [AIMessage(content=f"**Initiating Neurological Video Scan...**\n\n{instruction}")]
        }
        
    # 2. ANALYZE SDK RESULTS: The frontend completed the test and sent back the latency/velocity data!
    print(f"🧬 Analyzing SDK Physics Data: {test_results}")
    
    # Ask the LLM to interpret the raw physics data from the Vision SDK
    system_prompt = f"""
    You are a Neuro-Ophthalmologist AI. The frontend Vision Agents SDK just completed a {test_type} video scan and returned this raw data:
    {json.dumps(test_results, indent=2)}
    
    YOUR GOAL:
    Translate this raw latency/velocity data into a human-readable clinical assessment. 
    - If latency is >300ms for PLR, mention it's sluggish (possible concussion/trauma).
    - If Nystagmus micro-stutters are detected, mention vestibular concerns.
    Keep it highly empathetic and medical. DO NOT spit out raw JSON to the user.
    """
    
    analysis = llm.invoke([SystemMessage(content=system_prompt)])
    
    # Route them to a trauma/neurology protocol questionnaire next
    return {
        "video_stream_active": False,
        "active_protocol": "neurology_protocol", 
        "protocol_step": "start",
        "messages": [AIMessage(content=analysis.content)]
    }