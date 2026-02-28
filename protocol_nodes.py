import os
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage
from state import AgentState
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0.4, 
    api_key=os.getenv("GROQ_API_KEY")
)

# --- PROTOCOL DEFINITIONS ---
PROTOCOLS = {
    "conjunctivitis_protocol": {
        "questions": [
            "Do you have sticky yellow or green discharge?",
            "Is there a gritty sensation (like sand in your eye)?",
            "Are your eyelids stuck together when you wake up?",
            "Is the redness in one eye or both?",
            "Are you experiencing any itching or burning sensation?",
            "Do you have excessive tearing or watery eyes?",
            "Are you experiencing any sensitivity to light (photophobia)?",
            "Have you recently had a cold, or been around anyone else with pink eye?",
            "Do you currently wear contact lenses?",
            "Are you experiencing any blurriness or decrease in your actual vision?"
        ]
    },
    "sch_protocol": {
        "questions": [
            "Did this red spot appear after coughing, sneezing, vomiting, or heavy lifting?",
            "Are you experiencing any actual pain (other than a mild scratchy feeling)?",
            "Are you currently taking any blood thinners or aspirin?",
            "Did you experience any recent trauma, injury, or poke to the eye?",
            "Do you have a history of high blood pressure?",
            "Is the red patch affecting or blocking your vision in any way?",
            "Is there any discharge, crusting, or stickiness accompanying the redness?",
            "Do you have a habit of rubbing your eyes vigorously?",
            "Has this sudden redness happened to you before?",
            "Is the red spot actively growing larger, or staying about the same size?"
        ]
    },
    "pterygium_protocol": {
        "questions": [
            "Do you spend a lot of time outdoors in the sun, wind, or dusty environments?",
            "Is the fleshy growth extending toward the center of your eye and interfering with your vision?",
            "Does it feel like there is a foreign object stuck in your eye?",
            "Is the eye frequently red, irritated, or inflamed?",
            "Do your eyes feel exceptionally dry or itchy on a regular basis?",
            "Has this growth noticeably changed in size or shape recently?",
            "Do you regularly wear UV-protective sunglasses when outside?",
            "Are you experiencing any double vision or astigmatism symptoms?",
            "Approximately how long ago did you first notice this growth?",
            "Does the area cause a burning sensation, especially after being outdoors?"
        ]
    },
    "stye_protocol": {
        "questions": [
            "Is there a specific, painful lump or bump on the edge of your eyelid?",
            "How many days has this bump been present?",
            "Does your entire eyelid feel swollen and heavy?",
            "Is there a visible white or yellow 'head' (like a pimple) at the center of the bump?",
            "Is the lump extremely sensitive or tender when you touch it?",
            "Are you experiencing increased tearing or crustiness along the lash line?",
            "Is the swelling severe enough that it is affecting your ability to see clearly?",
            "Have you been applying warm compresses to the area yet?",
            "Have you had styes or similar eyelid issues in the past?",
            "Do you regularly wear eye makeup, or occasionally sleep without removing it?"
        ]
    },
    "neurology_protocol": {
        "questions": [
            "Have you recently experienced any head trauma, falls, or direct impact to your head?",
            "Are you feeling dizzy, lightheaded, or experiencing a spinning sensation (vertigo)?",
            "Are you experiencing any double vision, or blurry vision that changes when you move your head?",
            "Have you felt nauseous or vomited since these symptoms started?",
            "Are you currently taking any new medications, antihistamines, or substances that might affect your pupils?",
            "Do you have a severe headache or extreme sensitivity to light right now?",
            "Have you noticed any weakness, numbness, or tingling in your face, arms, or legs?",
            "Is there a history of migraines, strokes, or neurological conditions in your family?"
        ]
    }
}

def run_assessment_protocol(state: AgentState):
    """
    Runs the questionnaire step-by-step.
    At the end, it generates a DOCTOR-STYLE SUMMARY using LLM synthesis.
    """
    protocol_name = state.get("active_protocol")
    
    # "start" is the default trigger value from Triage
    step = state.get("protocol_step") 
    if step == "start":
        step = 0
    elif step is None:
        step = 0
    else:
        step = int(step)

    messages = state["messages"]
    
    # Safety Check
    if not protocol_name or protocol_name not in PROTOCOLS:
        return {"messages": [AIMessage(content="I'm having trouble retrieving the medical protocol.")], "active_protocol": None}

    questions = PROTOCOLS[protocol_name]["questions"]
    
    # --- LOGIC: ASK NEXT QUESTION ---
    # If we are just starting (step 0), we ask Q1.
    # If we are in the middle, we check if there are more questions.
    
    if step < len(questions):
        question_text = questions[step]
        
        # Add a prefix so the user knows where they are
        prefix = f"**Follow-up Question ({step + 1}/{len(questions)}):**\n\n"
        
        return {
            "protocol_step": step + 1,
            "messages": [AIMessage(content=f"{prefix}{question_text}")]
        }

    # --- FINAL STEP: GENERATE REPORT (The Fix) ---
    else:
        print(f"📝 Protocol {protocol_name} Finished. Generating Final Report...")
        
        # 1. Gather all Q&A from history
        # We assume the last (len(questions) * 2) messages are the Q&A pairs
        # This is a heuristic; for production, we might store answers in state explicitly.
        qa_history = messages[-(len(questions)*2):]
        history_text = "\n".join([f"{m.type.upper()}: {m.content}" for m in qa_history])
        
        disease_name = protocol_name.replace("_protocol", "").title()
        
        # 2. Ask LLM to synthesize (Not Parrot)
        system_prompt = f"""
        You are a Senior Ophthalmologist. You just finished asking a patient diagnostic questions for **{disease_name}**.
        
        CHAT HISTORY OF Q&A:
        {history_text}
        
        TASK:
        Write a final diagnosis report.
        1. **Do NOT list the questions and answers.** (e.g. Don't say "You said Yes to discharge").
        2. Instead, **Synthesize** the findings. (e.g. "Since you reported sticky discharge and waking up with glued eyes, this indicates Bacterial Conjunctivitis.")
        3. Provide **Actionable Advice** (Home care, when to see a doctor).
        4. Keep it professional but empathetic.
        
        OUTPUT FORMAT:
        ## 📋 Diagnosis Report: {disease_name}
        **Analysis:** [Your synthesis]
        **Recommendation:** [Your advice]
        """
        
        report = llm.invoke([SystemMessage(content=system_prompt)])
        
        # Reset protocol so the user can ask something else next
        return {
            "active_protocol": None, 
            "protocol_step": None,
            "messages": [AIMessage(content=report.content)]
        }