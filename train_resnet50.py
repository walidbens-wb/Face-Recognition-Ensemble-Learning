import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os, time, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ===================================================================
# CONFIGURATION : SNIPER RESNET-50 SUR DATASET AUGMENTÉ
# ===================================================================
MODEL_NAME_TO_TRAIN = "resnet50" # <-- On passe au 50 !
TRAIN_DIR = 'dataset_sniper_final/train' 
VAL_DIR   = 'dataset_sniper_final/val'
BEST_PATH = "resnet50_sniper.pt" # <-- Fichier de sortie
# ===================================================================

def main(model_name):
    BATCH_SIZE = 32 # On peut augmenter un peu car ResNet50 est moins lourd que 101
    EPOCHS = 40     # Moins d'époques car on a 10x plus d'images par époque
    LR = 1e-4       # Learning rate un peu plus doux
    NUM_WORKERS = 2
    SEED = 42

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device : {device} | Modèle : {model_name}")
    print(f"📂 Train sur : {TRAIN_DIR} (Images 1-6 multipliées)")
    print(f"📂 Val sur   : {VAL_DIR} (Image 7 uniquement)")

    if not os.path.exists(TRAIN_DIR):
        print(f"❌ ERREUR : Lance d'abord le script 'generate_data.py' !")
        return

    # --- Data Augmentation ---
    # On garde une augmentation forte même si on a déjà augmenté les données
    # pour éviter l'overfitting à tout prix.
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # Zoom léger
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        # On garde ton jitter agressif pour les images noires
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1)
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean, std)
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_tf)
    
    print(f"📊 Images Train : {len(train_ds)}")
    print(f"📊 Images Val   : {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    num_classes = len(train_ds.classes)

    # --- Chargement du Modèle (ResNet-50) ---
    print(f"🏗️ Chargement de {model_name}...")
    
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # On dégèle les couches profondes pour affiner les visages
        for name, param in model.named_parameters():
             if "layer3" in name or "layer4" in name or "fc" in name:
                 param.requires_grad = True
             else:
                 param.requires_grad = False
        print("🔒 ResNet-50 : Gelé sauf layer3, layer4, fc.")

    else:
        # Fallback ResNet101 si tu veux vraiment
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    model.to(device)

    # --- Optimisation ---
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3, verbose=True)

    # Boucle
    best_acc = 0.0
    
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
            optimizer.step()
            tr_correct += (outputs.argmax(1) == labels).sum().item()
            tr_total += imgs.size(0)
            
            # Mise à jour de la barre de progression
            loop.set_postfix(acc=tr_correct/tr_total)
            
        train_acc = tr_correct / tr_total

        # Validation (Sur l'image 7)
        model.eval()
        val_correct, val_total = 0, 0
        with torch.inference_mode():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                val_correct += (model(imgs).argmax(1) == labels).sum().item()
                val_total += imgs.size(0)
        val_acc = val_correct / val_total
        
        scheduler.step(val_acc)
        print(f"Epoch {epoch:02d} | Tr Acc: {train_acc:.4f} | Val Acc (img 7): {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": train_ds.classes}, BEST_PATH)
            print(f"💾 SAUVEGARDÉ : {best_acc:.4f}")

    print(f"🏁 Terminé. Meilleur Sniper (ResNet50) : {best_acc:.4f}")

if __name__ == "__main__":
    main(MODEL_NAME_TO_TRAIN)