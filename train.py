import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import copy
import time
import random

# --- Configuration ---
DATA_ROOT = r"Artistick"  # Relative path to data root
AI_DIR = os.path.join(DATA_ROOT, "AiArtData", "AiArtData")
REAL_DIR = os.path.join(DATA_ROOT, "RealArt", "RealArt")

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
PATIENCE = 5
NUM_WORKERS = 0 # Windows often needs 0 workers for stability
DEVICE = torch.device("cpu") # User requested CPU only

print(f"Using device: {DEVICE}")

# --- Custom Dataset ---
class AiRealDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy tensor or handle error appropriately
            # For simplicity, returning a zero tensor and label -1 (to be ignored or handled)
            return torch.zeros((3, 224, 224)), label

# --- Augmentations ---
class RandomJPEG(object):
    def __init__(self, quality_min=75, quality_max=90):
        self.quality_min = quality_min
        self.quality_max = quality_max

    def __call__(self, img):
        if random.random() < 0.5: # Apply 50% of the time
            return img
            
        quality = random.randint(self.quality_min, self.quality_max)
        img_buffer = os.path.join("temp_aug.jpg") # fast temp file (or use io.BytesIO)
        
        # Using functional JPEG compression if possible, or PIL directly
        # For simplicity/correctness with PIL images:
        try:
            from io import BytesIO
            buffer = BytesIO()
            img.save(buffer, "JPEG", quality=quality)
            buffer.seek(0)
            return Image.open(buffer).convert("RGB")
        except:
            return img
            
    def __repr__(self):
        return self.__class__.__name__ + f'(quality_min={self.quality_min}, quality_max={self.quality_max})'

train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    RandomJPEG(75, 90),
    transforms.RandomAdjustSharpness(2, p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Data Preparation ---
def get_data_loaders():
    # 1. Collect file paths
    # Note: Using glob to match common image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
    
    ai_files = []
    for ext in extensions:
        ai_files.extend(glob.glob(os.path.join(AI_DIR, ext)))
        ai_files.extend(glob.glob(os.path.join(AI_DIR, ext.upper())))
        
    real_files = []
    for ext in extensions:
        real_files.extend(glob.glob(os.path.join(REAL_DIR, ext)))
        real_files.extend(glob.glob(os.path.join(REAL_DIR, ext.upper())))

    print(f"Found {len(ai_files)} AI images and {len(real_files)} Real images.")
    
    if len(ai_files) == 0 or len(real_files) == 0:
        raise ValueError("Could not find images. Check paths!")

    # 2. FLATTEN LISTS & LABELS: 0 for Real, 1 for AI (typical convention: 0 is negative/normal, 1 is positive/detected)
    # Let's define: 0 = Real, 1 = AI
    all_files = real_files + ai_files
    all_labels = [0] * len(real_files) + [1] * len(ai_files)

    # 3. Calculate Class Weights
    count_0 = len(real_files)
    count_1 = len(ai_files)
    total = count_0 + count_1
    
    # Weight = Total / (2 * Count)
    weight_0 = total / (2 * count_0)
    weight_1 = total / (2 * count_1)
    class_weights = torch.FloatTensor([weight_0, weight_1]).to(DEVICE)
    print(f"Class Weights -> Real (0): {weight_0:.4f}, AI (1): {weight_1:.4f}")

    # 4. Split
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    # 5. Datasets & Loaders
    train_dataset = AiRealDataset(train_files, train_labels, transform=train_transforms)
    val_dataset = AiRealDataset(val_files, val_labels, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    return train_loader, val_loader, class_weights

# --- Model Setup ---
def build_model():
    print("Loading MobileNetV3-Large...")
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    
    # FREEZE BACKBONE
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace Classifier Head
    # Original classifier: Sequential(Linear(960, 1280), Hardswish(), Dropout(p=0.2, inplace=True), Linear(1280, 1000))
    # We want to keep the penultimate structure if possible, or just replace the final linear.
    # MobileNetV3 classifier structure in torchvision:
    # (0): Linear(in_features=960, out_features=1280, bias=True)
    # (1): Hardswish()
    # (2): Dropout(p=0.2, inplace=True)
    # (3): Linear(in_features=1280, out_features=1000, bias=True)
    
    # We will replace the last fully connected layer (3)
    num_ftrs = model.classifier[3].in_features # Should be 1280
    model.classifier[3] = nn.Linear(num_ftrs, 2)
    
    # Ensure the new head is trainable (it is by default on creation, but good to double check flow)
    for param in model.classifier[3].parameters():
        param.requires_grad = True
        
    return model.to(DEVICE)

# --- Training Loop ---
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20, start_epoch=0, initial_best_acc=0.0):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = initial_best_acc
    best_loss = float('inf')
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    epochs_no_improve = 0
    
    # Adjust loop range
    for epoch in range(start_epoch, epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            # Deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # SAVE CHECKPOINT IMMEDIATELY (Full state)
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_acc': best_acc
                    }
                    torch.save(checkpoint, "best_model.pth")
                    print(f"  * New best model saved! (Acc: {best_acc:.4f})")
                
                # Early Stopping check (based on LOSS or ACC? usually Loss is better for stability)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
        
        print()
        
        if epochs_no_improve >= PATIENCE:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

if __name__ == '__main__':
    try:
        # 1. Load Data
        train_loader, val_loader, class_weights = get_data_loaders()
        
        # 2. Build Model
        model = build_model()
        
        # 3. Define Loss & Optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # --- RESUME LOGIC ---
        start_epoch = 0
        best_acc = 0.0
        checkpoint_path = "best_model.pth"
        
        if os.path.exists(checkpoint_path):
            print(f"Checkout '{checkpoint_path}' found. Attempting to resume...")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                
                # Check if it's a legacy checkpoint (just weights) or new checkpoint (dict)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # New format
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if 'epoch' in checkpoint:
                        start_epoch = checkpoint['epoch'] + 1 # Start from next epoch
                    if 'best_acc' in checkpoint:
                        best_acc = checkpoint['best_acc']
                    print(f"Resumed from epoch {start_epoch} with Best Acc: {best_acc:.4f}")
                else:
                    # Legacy format (weights only)
                    model.load_state_dict(checkpoint)
                    print("loaded legacy model weights. Starting from epoch 0 (optimizer reset).")
                    
            except Exception as e:
                print(f"Failed to load checkpoint: {e}. Starting from scratch.")

        # 4. Train
        # We need to pass start_epoch and best_acc to train_model to resume correctly
        model, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=EPOCHS, start_epoch=start_epoch, initial_best_acc=best_acc)
        
        # 5. Save Final Model (although best is saved during training)
        # torch.save(model.state_dict(), "final_model.pth")
        print("Training finished.")

        # 6. Save simple plot (ASCII or print)
        print("Done.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
