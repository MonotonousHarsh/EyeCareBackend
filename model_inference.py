import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import numpy as np

# --- CONFIGURATION ---
# Ensure this path points to your actual .pth file (not a folder!)
MODEL_PATH = "/home/harsh/Documents/C-drive/AI-Stuff(Agents,Chatbot)/Dic X SC/Chat-bot/backend/best_vit_base_4class.pth"

CLASSES = ['healthy', 'pterygium', 'conjunctivitis','pinguecula'] # Mapped 0, 1, 2
NUM_CLASSES = len(CLASSES) # <--- FIXED: Defined this variable so the function can use it
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL DEFINITION (Must match training exactly) ---
def create_vit():
    # We use num_classes=0 to get features, then add our own head
    # pretrained=False because we are loading our own weights next
    model = timm.create_model("vit_base_patch16_224", pretrained=False) 
    in_features = model.head.in_features
    
    model.head = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, NUM_CLASSES) # Now this variable exists!
    )
    return model

# --- LOAD MODEL (Global Instance) ---
print(f"🔄 Loading AI Model on {DEVICE}...")
try:
    model = create_vit()
    
    if os.path.exists(MODEL_PATH):
        # Load weights
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Model weights loaded successfully.")
    else:
        print(f"⚠️ WARNING: {MODEL_PATH} not found. Predictions will be random!")
        # We don't exit here so the server doesn't crash, but it won't predict correctly without weights.
    
    model.to(DEVICE)
    model.eval()
    
except Exception as e:
    print(f"❌ Model Load Error: {e}")
    model = None

# --- PREPROCESSING ---
# Same validation transforms as your training script
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- INFERENCE FUNCTION ---
def predict_eye_disease(image_path: str):
    """
    Input: Path to an image file.
    Output: Dictionary {"class": str, "confidence": float}
    """
    if model is None:
        return {"class": "error", "confidence": 0.0}
    
    try:
        # 1. Load Image
        img = Image.open(image_path).convert("RGB")
        
        # 2. Transform
        img_tensor = val_transform(img).unsqueeze(0).to(DEVICE) # Add batch dimension
        
        # 3. Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top class
            top_prob, top_class_id = probabilities.max(1)
            
            confidence = top_prob.item()
            class_name = CLASSES[top_class_id.item()]
            
            return {"class": class_name, "confidence": confidence}
            
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return {"class": "error", "confidence": 0.0}

# Test block (Runs only if you execute this file directly)
if __name__ == "__main__":
    # Create a dummy image to test
    dummy = Image.new('RGB', (224, 224), color='red')
    dummy.save("test.jpg")
    print("Testing with dummy image...")
    print(predict_eye_disease("test.jpg"))