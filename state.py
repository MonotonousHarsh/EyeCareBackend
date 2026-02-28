import operator
from typing import TypedDict, List, Annotated, Optional, Dict, Any
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class UserProfile(TypedDict):
    age: int
    gender: str
    city: str          # Critical for finding doctors
    medical_history: List[str]
    name: str

class VisionTestResults(TypedDict):
    distance_acuity_left: Optional[str]
    distance_acuity_right: Optional[str]
    near_vision_score: Optional[str]
    color_blindness: Optional[bool]
    astigmatism_detected: Optional[bool]
    last_test_date: Optional[str]

class AgentState(TypedDict):
    # 1. CONVERSATION HISTORY
    messages: Annotated[List[AnyMessage],add_messages]
    
    # 2. USER CONTEXT
    user_profile: Optional[UserProfile]
    vision_test_results: Optional[VisionTestResults]
    
    # 3. VISION & DIAGNOSIS STATE
    uploaded_image_id: Optional[str]
    video_stream_active: Optional[bool]
    functional_test_type: Optional[str]
    functional_test_results: Optional[Dict[str, Any]]
    ai_prediction: Optional[Dict[str, Any]] 
    # Example: {"class": "pterygium", "confidence": 0.98}

    # 4. PROTOCOL STATE (The Questionnaire Engine)
    active_protocol: Optional[str] 
    # Options: "sch_protocol", "stye_protocol", "pterygium_protocol", "conjunctivitis_protocol"
    
    protocol_step: Optional[str] 
    # Tracks the current question: "start", "q1_pain", "q2_vision"
    
    assessment_data: Optional[Dict[str, Any]]
    # Stores answers: {"has_pain": True, "vision_affected": False}
    
    # 5. BRAIN STATE (Routing & Booking)
    intent: Optional[str]             # e.g. "medical_diagnosis", "book_appointment"
    detected_condition: Optional[str] # e.g., "Glaucoma", "Myopia"
    selected_doctor_id: Optional[int] # ID of the doctor user wants to book