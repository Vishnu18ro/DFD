import os
import shutil
import random
import glob

# Configuration
DATA_ROOT = r"Artistick"
AI_DIR = os.path.join(DATA_ROOT, "AiArtData", "AiArtData")
REAL_DIR = os.path.join(DATA_ROOT, "RealArt", "RealArt")
DEST_DIR = r"static/recommendations"

# Ensure destination exists
os.makedirs(DEST_DIR, exist_ok=True)

# Function to get images
def get_images(directory, limit=50):
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
        files.extend(glob.glob(os.path.join(directory, ext.upper())))
    
    # Shuffle and pick
    random.shuffle(files)
    return files[:limit]

# Copy images
def copy_images(files, prefix):
    count = 0
    for src in files:
        try:
            filename = os.path.basename(src)
            # Create a unique name to avoid overwrites and identify type
            dest_name = f"{prefix}_{filename}"
            dest_path = os.path.join(DEST_DIR, dest_name)
            shutil.copy2(src, dest_path)
            count += 1
            print(f"Copied {filename} to {dest_name}")
        except Exception as e:
            print(f"Error copying {src}: {e}")
    return count

if __name__ == "__main__":
    # Clear existing
    if os.path.exists(DEST_DIR):
        for f in os.listdir(DEST_DIR):
            os.remove(os.path.join(DEST_DIR, f))

    print("Selecting Real images...")
    real_images = get_images(REAL_DIR, 50)
    print(f"Selected {len(real_images)} real images.")
    
    print("Selecting AI images...")
    ai_images = get_images(AI_DIR, 50)
    print(f"Selected {len(ai_images)} AI images.")
    
    # Copy
    real_count = copy_images(real_images, "real")
    ai_count = copy_images(ai_images, "ai")
    
    print(f"\nDone! Copied {real_count} real and {ai_count} AI images to {DEST_DIR}.")
