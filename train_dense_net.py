import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os, time, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ===================================================================
# CONFIGURATION : DENSENET-121 (Inversé : Train sur 7, Val sur 6)
# ===================================================================
MODEL_NAME = "densenet121"
TRAIN_DIR = 'dataset_inverted/train' # <-- Pointe bien sur le nouveau dataset
VAL_DIR   = 'dataset_inverted/val'
BEST_PATH = "densenet121_sniper.pt"
# ===================================================================

def main():
    BATCH_SIZE = 32
    EPOCHS = 40
    LR = 1e-4
    SEED = 42

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device : {device} | Modèle : {MODEL_NAME} (INVERSÉ)")

    if not os.path.exists(TRAIN_DIR):
        print("❌ Erreur : Dossier dataset_inverted introuvable.")
        return

    # Augmentations identiques aux autres pour la cohérence
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(), transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1)
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean, std)
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_tf)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"📊 Train (sur img 7 inclus): {len(train_ds)} | Val (sur img 6): {len(val_ds)}")

    # --- Chargement DenseNet-121 ---
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, len(train_ds.classes))

    # Stratégie Dégel spécifique DenseNet
    # On dégèle le dernier bloc dense et le classifier
    for name, param in model.named_parameters():
        if 'denseblock4' in name or 'norm5' in name or 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    print("🔒 DenseNet-121 : Gelé sauf dernier bloc.")
    model.to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=4, verbose=True)

    best_acc = 0.0
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

        # Validation (Sur Image 6 !)
        model.eval()
        val_correct, val_total = 0, 0
        with torch.inference_mode():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                val_correct += (model.forward(imgs).argmax(1) == labels).sum().item()
                val_total += imgs.size(0)
        val_acc = val_correct / val_total
        
        scheduler.step(val_acc)
        print(f"Epoch {epoch:02d} | Tr: {train_acc:.4f} | Val (img 6): {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": train_ds.classes}, BEST_PATH)
            print(f"💾 NEW BEST: {best_acc:.4f}")

    print(f"🏁 Terminé. DenseNet-121 Inversé : {best_acc:.4f}")

if __name__ == "__main__":
    main()