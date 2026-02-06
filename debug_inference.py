import torch
from inference import load_model, predict_image
import os
import traceback

def test_inference():
    print("Testing inference...")
    image_path = r"d:\projects\DFD\uploads\047fc6ddcac76293_OIP.webp"
    model_path = r"d:\projects\DFD\best_model.pth"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
        
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    try:
        print("Loading model...")
        model = load_model(model_path)
        print("Model loaded.")
        
        print(f"Predicting on {image_path}...")
        label, score = predict_image(model, image_path)
        print(f"Result: {label}, Score: {score}")
        
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    test_inference()
