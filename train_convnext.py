import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os, time, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ===================================================================
# CONFIGURATION : CONVNEXT GLOBAL (Le "Patron" du Test-7)
# ===================================================================
MODEL_NAME = "convnext_small"
TRAIN_DIR = 'dataset_convnext_full/train'
VAL_DIR   = 'dataset_convnext_full/val'
BEST_PATH = "convnext_global.pt" 
# ===================================================================

def main():
    # ConvNeXt est lourd, baisse le BATCH_SIZE si tu as une erreur de mémoire (VRAM)
    BATCH_SIZE = 16 
    EPOCHS = 30 
    LR = 1e-5
    SEED = 42

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device : {device} | Entraînement : {MODEL_NAME} (GLOBAL)")
    print(f"📂 Train sur : {TRAIN_DIR}")
    print(f"📂 Val sur   : {VAL_DIR}")

    if not os.path.exists(TRAIN_DIR):
        print("❌ ERREUR : Le dossier dataset_convnext_full n'existe pas.")
        return

    # --- AUGMENTATIONS SPÉCIALES "GLOBAL + ROTATION" ---
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    train_tf = transforms.Compose([
        # 1. On redimensionne en gardant les proportions (petit côté = 256)
        transforms.Resize(256), 
        # 2. On prend un carré de 224 au hasard (voit les vêtements + tête)
        transforms.RandomCrop(224),    
        
        transforms.RandomHorizontalFlip(),
        
        # 3. ROTATION ENCORE : On insiste pour qu'il soit robuste
        transforms.RandomRotation(degrees=45), 
        
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
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
    
    # Num workers = 0 si tu es sous Windows et que ça plante, sinon 2
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"📊 Images Train : {len(train_ds)} | Images Val : {len(val_ds)}")
    print(f"📊 Classes : {len(train_ds.classes)}")

    # --- CHARGEMENT DU MODÈLE ---
    print(f"🏗️ Chargement de {MODEL_NAME}...")
    model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
    
    # Remplacement de la dernière couche pour nos 114 classes
    # ConvNeXt a une structure différente : classifier[2] est la finale
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, len(train_ds.classes))

    # ON DÉGÈLE TOUT (Full Fine-Tuning)
    # On veut qu'il réapprenne à voir les formes (vêtements, corps couchés)
    for param in model.parameters():
        param.requires_grad = True
            
    model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_acc = 0.0
    start_time = time.time()
    
    # --- BOUCLE D'ENTRAÎNEMENT ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_correct, tr_total = 0, 0
        loop = tqdm(train_loader, leave=False, desc=f"Epoch {epoch}/{EPOCHS}")
        
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Clip grad pour éviter les explosions
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            tr_correct += (outputs.argmax(1) == labels).sum().item()
            tr_total += imgs.size(0)
            loop.set_postfix(acc=tr_correct/tr_total)
            
        train_acc = tr_correct / tr_total

        # Validation
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

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": train_ds.classes}, BEST_PATH)
            print(f"💾 NEW BEST: {best_acc:.4f}")

    duration = (time.time() - start_time) / 60
    print(f"🏁 Terminé. ConvNeXt Global : {best_acc:.4f} en {duration:.1f} min")

if __name__ == "__main__":
    main()