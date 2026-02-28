from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage
from state import AgentState
from db_mock import fetch_user_data

# Import core nodes
from nodes import (
    triage_node, 
    medical_advice_node, 
    find_doctor_node, 
    booking_node, 
    general_chat_node
)

# IMPORTS UPDATED: We now have two distinct vision nodes
from vision_nodes import (
    structural_vision_analysis_node,
    functional_vision_analysis_node
)
from protocol_nodes import run_assessment_protocol

# --- LOADER ---
def load_user_context(state: AgentState):
    if not state.get("user_profile"):
        data = fetch_user_data("user_123")
        if data:
            return {
                "user_profile": data["profile"], 
                "vision_test_results": data["test_results"]
            }
    return {}

# --- ROUTER 1: INTELLIGENT ROUTER EXECUTOR ---
def main_router(state: AgentState):
    """
    Simply routes to the node decided by the Intelligent Triage Supervisor.
    """
    next_node = state.get("intent")
    print(f"🔀 EXECUTING ROUTE: Going to -> '{next_node}'")
    
    # Mapped to our new Hackathon paths
    if next_node == "assessment_protocol": return "assessment_protocol"
    if next_node == "structural_vision_analysis": return "structural_vision_analysis"
    if next_node == "functional_vision_analysis": return "functional_vision_analysis"
    if next_node == "medical_advice": return "medical_advice"
    if next_node == "find_doctor": return "find_doctor"
    if next_node == "booking": return "booking"
        
    # Default fallback
    return "general_chat"

# --- ROUTER 2: POST-VISION ROUTER ---
def vision_router(state: AgentState):
    """
    Determines if we need to start a protocol immediately after ANY vision analysis.
    """
    if state.get("active_protocol"):
        return "assessment_protocol"
    return "end"

# --- GRAPH DEFINITION ---
builder = StateGraph(AgentState)

# 1. Add All Nodes (Goodbye request_image_node!)
builder.add_node("load_context", load_user_context)
builder.add_node("triage", triage_node)
builder.add_node("structural_vision_analysis", structural_vision_analysis_node)
builder.add_node("functional_vision_analysis", functional_vision_analysis_node)
builder.add_node("assessment_protocol", run_assessment_protocol)
builder.add_node("medical_advice", medical_advice_node)
builder.add_node("find_doctor", find_doctor_node)
builder.add_node("booking", booking_node)
builder.add_node("general_chat", general_chat_node)

# 2. Define Start Flow
builder.add_edge(START, "load_context")
builder.add_edge("load_context", "triage")

# 3. Define Main Routing Logic
builder.add_conditional_edges(
    "triage",
    main_router,
    {
        "assessment_protocol": "assessment_protocol",
        "structural_vision_analysis": "structural_vision_analysis",
        "functional_vision_analysis": "functional_vision_analysis",
        "medical_advice": "medical_advice",
        "find_doctor": "find_doctor",
        "booking": "booking",
        "general_chat": "general_chat"
    }
)

# 4. Define Post-Vision Logic (Both vision nodes route to the same evaluation router)
builder.add_conditional_edges(
    "structural_vision_analysis",
    vision_router,
    {"assessment_protocol": "assessment_protocol", "end": END}
)

builder.add_conditional_edges(
    "functional_vision_analysis",
    vision_router,
    {"assessment_protocol": "assessment_protocol", "end": END}
)

# 5. Define Other Endings
builder.add_edge("assessment_protocol", END)
builder.add_edge("medical_advice", END)
builder.add_edge("find_doctor", END)
builder.add_edge("booking", END)
builder.add_edge("general_chat", END)

# 6. Compile with Memory
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    # Draw the new architecture immediately!
    image_data = graph.get_graph().draw_mermaid_png()
    with open("bot_architecture_v2.png", "wb") as f:
        f.write(image_data)
        
    print("✅ Graph compiled successfully with Real-Time Video Routing! Saved as bot_architecture_v2.png")