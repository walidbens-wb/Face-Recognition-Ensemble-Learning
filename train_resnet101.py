import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os, time, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ===================================================================
# CONFIGURATION : RESNET-101 GLOBAL UPGRADE (Le Maillon Faible)
# ===================================================================
MODEL_NAME = "resnet101"
TRAIN_DIR = 'dataset_convnext_full/train' 
VAL_DIR   = 'dataset_convnext_full/val' # Image 6
BEST_PATH = "resnet101_global_upgraded.pt" 
PATIENCE_LIMIT = 10
# ===================================================================

def main():
    BATCH_SIZE = 16 
    EPOCHS = 40
    LR = 5e-5 
    SEED = 42

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device : {device} | Entraînement : {MODEL_NAME} (GLOBAL UPGRADE)")

    if not os.path.exists(TRAIN_DIR):
        print("❌ ERREUR : Dossier dataset_convnext_full introuvable.")
        return

    # --- AUGMENTATIONS SPÉCIALES "GLOBAL + ROTATION" ---
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    train_tf = transforms.Compose([
        transforms.Resize(256), 
        transforms.RandomCrop(224),    
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=45), # Rotation pour l'apprentissage
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1)
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_tf)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # --- MODÈLE ---
    print(f"🏗️ Chargement de {MODEL_NAME}...")
    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))

    # UNFREEZE TOUT 
    for param in model.parameters():
        param.requires_grad = True
            
    model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4) 
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_correct, tr_total = 0, 0
        loop = tqdm(train_loader, leave=False, desc=f"Ep {epoch}")
        
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            tr_correct += (outputs.argmax(1) == labels).sum().item()
            tr_total += imgs.size(0)
            loop.set_postfix(acc=tr_correct/tr_total)
            
        train_acc = tr_correct / tr_total

        # Validation (Sur Image 6)
        model.eval()
        val_correct, val_total = 0, 0
        with torch.inference_mode():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                val_correct += (model(imgs).argmax(1) == labels).sum().item()
                val_total += imgs.size(0)
        val_acc = val_correct / val_total
        
        scheduler.step(val_acc)
        print(f"Epoch {epoch:02d} | Tr Acc: {train_acc:.4f} | Val Acc (Img 6): {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # --- EARLY STOPPING ---
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0 
            torch.save({"model": model.state_dict(), "classes": train_ds.classes}, BEST_PATH)
            print(f"💾 NEW BEST: {best_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE_LIMIT:
                print("\n🛑 EARLY STOPPING : Arrêt.")
                break

if __name__ == "__main__":
    main()  