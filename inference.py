import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os
import argparse

# --- Configuration ---
DEVICE = torch.device("cpu")

# --- Model Loading ---
def load_model(model_path="best_model.pth"):
    print(f"Loading model from {model_path}...")
    model = models.mobilenet_v3_large(weights=None) # No need to download weights, we load state dict
    
    # Recreate the head structure to match training
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, 2)
    
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Have you trained the model yet?")
        sys.exit(1)
        
    model.to(DEVICE)
    model.eval()
    return model

# --- Preprocessing ---
# Must match Validation transforms from train.py
inference_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(model, image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = inference_transforms(image).unsqueeze(0).to(DEVICE) # Add batch dimension
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
        # Class 0: Real, Class 1: AI (Based on alphabetical order in training or explicit mapping)
        # Training script explicit mapping: 0=Real, 1=AI
        
        label = "AI-Generated" if predicted_class.item() == 1 else "Real Image"
        score = confidence.item() * 100
        
        return label, score
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Image Detector Inference")
    parser.add_argument("--image", type=str, help="Path to a single image file")
    parser.add_argument("--dir", type=str, help="Path to a directory of images")
    parser.add_argument("--model", type=str, default="best_model.pth", help="Path to the trained model file")
    
    args = parser.parse_args()
    
    if not args.image and not args.dir:
        print("Please provide --image or --dir")
        sys.exit(1)
        
    model = load_model(args.model)
    
    print("-" * 30)
    print(f"{'PREDICTION':<15} | {'CONFIDENCE':<10} | {'FILENAME'}")
    print("-" * 30)

    if args.image:
        label, score = predict_image(model, args.image)
        if label:
            print(f"{label:<15} | {score:.2f}%     | {os.path.basename(args.image)}")

    if args.dir:
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        files = [f for f in os.listdir(args.dir) if os.path.splitext(f)[1].lower() in valid_exts]
        
        for f in files:
            path = os.path.join(args.dir, f)
            label, score = predict_image(model, path)
            if label:
                print(f"{label:<15} | {score:.2f}%     | {f}")
    
    print("-" * 30)
