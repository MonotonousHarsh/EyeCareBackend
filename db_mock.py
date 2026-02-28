from state import UserProfile, VisionTestResults
from typing import List, Dict

# --- MOCK DATABASE OF USERS ---
def fetch_user_data(user_id: str):
    if user_id == "user_123":
        return {
            "profile": UserProfile(
                name="Rahul",
                age=45,
                gender="Male",
                city="Agra",  # Added City for location-based search
                medical_history=["diabetes"]
            ),
            "test_results": VisionTestResults(
                distance_acuity_left="20/60",
                distance_acuity_right="20/20",
                near_vision_score="N6",
                color_blindness=False,
                astigmatism_detected=True,
                last_test_date="2024-05-20"
            )
        }
    return None

# --- MOCK DATABASE OF DOCTORS (Simulating Google Maps/Zocdoc) ---
# We include "tags" to match against specific diseases (e.g., Glaucoma, Retina)
MOCK_DOCTORS = [
    {
        "id": 101, 
        "name": "Dr. Aditi Sharma", 
        "clinic": "Agra Eye Centre",
        "city": "Agra", 
        "specialties": ["Glaucoma", "General Ophthalmology"], 
        "rating": 4.9, 
        "next_available": "Tomorrow, 10:00 AM"
    },
    {
        "id": 102, 
        "name": "Dr. Rajesh Verma", 
        "clinic": "Verma Retina Clinic",
        "city": "Agra", 
        "specialties": ["Retina", "Diabetic Retinopathy"], 
        "rating": 4.5, 
        "next_available": "Today, 4:00 PM"
    },
    {
        "id": 103, 
        "name": "Dr. Simran Kaur", 
        "clinic": "Chandigarh Vision Care",
        "city": "Chandigarh", 
        "specialties": ["Pediatric", "General"], 
        "rating": 4.8, 
        "next_available": "Tomorrow, 9:00 AM"
    }
]

def fetch_doctors(city: str, condition_keyword: str = None) -> List[Dict]:
    """
    Simulates searching for a doctor.
    - city: Filter by location.
    - condition_keyword: If user has 'Glaucoma', prioritize Glaucoma specialists.
    """
    results = []
    for doc in MOCK_DOCTORS:
        # 1. Location Filter (Strict)
        if doc["city"].lower() != city.lower():
            continue
            
        # 2. Specialty Boosting (Soft Filter)
        # If we need a Retina specialist, checking if this doctor handles it
        if condition_keyword:
            # Simple keyword matching for simulation
            matches = any(condition_keyword.lower() in s.lower() for s in doc["specialties"])
            if matches:
                doc["relevance"] = "High"
            else:
                doc["relevance"] = "Medium"
        
        results.append(doc)
    
    # Sort by relevance then rating
    return sorted(results, key=lambda x: x.get("relevance", "Low"), reverse=True)