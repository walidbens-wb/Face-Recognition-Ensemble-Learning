

class EnsembleModel:
    def __init__(self, models_list, device):
        self.models = models_list
        self.device = device

    def eval(self):
        for m in self.models: m.eval()

    def to(self, device):
        for m in self.models: m.to(device)
        self.device = device
        return self

def build_model(torch, nn, models, classes=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_list = []
    final_classes = None

    def load_layer(name, model_fn, fc_attr):
        try:
            ckpt = torch.load(name, map_location="cpu")
            cls_list = [str(c) for c in ckpt["classes"]]
            m = model_fn(weights=None)
            
            # Gestion Architecture ConvNeXt
            if fc_attr == "classifier_convnext":
                in_features = m.classifier[2].in_features
                m.classifier[2] = nn.Linear(in_features, len(cls_list))
            elif fc_attr == "classifier": 
                # EffNet/DenseNet
                if isinstance(m.classifier, nn.Sequential):
                    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(cls_list))
                else:
                    m.classifier = nn.Linear(m.classifier.in_features, len(cls_list))
            else:
                m.fc = nn.Linear(m.fc.in_features, len(cls_list))

            m.load_state_dict(ckpt["model"], strict=True)
            m.to(device).eval()
            print(f"✅ {name} chargé")
            return m, cls_list
        except Exception as e: 
            return None, None

    # --- LE QUATUOR GAGNANT ---
    # 1. ConvNeXt Global (Leader - Index 0)
    m, c = load_layer("convnext_global.pt", models.convnext_small, "classifier_convnext")
    if m: models_list.append(m); final_classes = c

    # 2. ResNet-50 Sniper (Robustesse)
    m, c = load_layer("resnet50_sniper.pt", models.resnet50, "fc")
    if m: models_list.append(m)

    # 3. DenseNet Inversé (Diversité)
    m, c = load_layer("densenet121_sniper.pt", models.densenet121, "classifier")
    if m: models_list.append(m)
    
    # 4. ResNet-101 Sniper (Mémoire)
    m, c = load_layer("resnet101_sniper.pt", models.resnet101, "fc")
    if m: models_list.append(m)
    
    if not models_list: 
        dummy = models.resnet18(weights=None)
        dummy.fc = nn.Linear(dummy.fc.in_features, 2)
        return dummy, ["0", "1"]
    
    return EnsembleModel(models_list, device), final_classes

# --- OUTILS ---

def manual_resize_aspect(img, target_size=256):
    w, h = img.size
    scale = target_size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), resample=2)

def manual_center_crop_fn(img, output_size=224):
    w, h = img.size
    left = (w - output_size) // 2
    top = (h - output_size) // 2
    return img.crop((left, top, left + output_size, top + output_size))

def manual_zoom_crop(img, zoom_factor=0.75):
    w, h = img.size
    crop_size = min(w, h) * zoom_factor
    left = (w - crop_size) / 2
    top = (h - crop_size) / 2
    right = (w + crop_size) / 2
    bottom = (h + crop_size) / 2
    return img.crop((left, top, right, bottom))

def get_majority_vote(votes):
    if not votes: return 0
    counts = {}
    for v in votes: counts[v] = counts.get(v, 0) + 1
    max_count = 0
    winner = votes[0]
    for v, c in counts.items():
        if c > max_count: max_count = c; winner = v
    return winner

def predict(ensemble, image, preprocess, torch):
    if not isinstance(ensemble, EnsembleModel):
        dummy = EnsembleModel([ensemble], next(ensemble.parameters()).device)
        ensemble = dummy

    if image.mode != "RGB": image = image.convert("RGB")
    device = ensemble.device
    
    to_tensor_tf = preprocess.transforms[1]
    normalize_tf = preprocess.transforms[2]

    # --- VUES ---
    img_256 = manual_resize_aspect(image, 256)
    img_global = manual_center_crop_fn(img_256, 224)
    
    def get_crop(z):
        return manual_center_crop_fn(manual_resize_aspect(manual_zoom_crop(image, z), 256), 224)
    c_std = get_crop(0.75)
    c_wide = get_crop(0.90)

    # --- BATCHS TTA LOURD ---
    # Pour le ConvNeXt Global (Index 0), on donne la vue globale.
    aug_global = [
        (img_global, False, 1.0), (img_global, True, 1.0), # Standard TTA
        (img_global, False, 0.5) # Gamma pour le Global (rotation apprise en train)
    ]
    
    # Pour les Snipers (1, 2, 3), on donne la vue face.
    aug_sniper = [
        (c_std, False, 1.0), 
        (c_std, True, 1.0),
        (c_wide, False, 1.0), 
        (c_std, False, 0.5) 
    ]

    # --- CONVERSION ---
    def batchify(configs):
        tensors = []
        for (pil, flip, gamma) in configs:
            if flip: curr = pil.transpose(0)
            else: curr = pil
            t = to_tensor_tf(curr)
            if gamma != 1.0: t = torch.pow(t, gamma)
            t = torch.clamp(t, 0.0, 1.0)
            tensors.append(normalize_tf(t))
        return torch.stack(tensors).to(device)

    batch_g = batchify(aug_global)
    batch_s = batchify(aug_sniper)

    # --- INFERENCE (Démocratie Pondérée) ---
    # Poids Égalitaires pour la diversité maximale
    weights = [0.25, 0.25, 0.25, 0.25]
    if len(ensemble.models) != 4:
        weights = [1.0/len(ensemble.models)] * len(ensemble.models)

    total_probs = None
    all_votes = []

    with torch.no_grad():
        for i, model in enumerate(ensemble.models):
            if i == 0: # ConvNeXt Global
                logits = model(batch_g)
                nb_aug = len(aug_global)
            else: # Snipers
                logits = model(batch_s)
                nb_aug = len(aug_sniper)

            probs = torch.nn.functional.softmax(logits, dim=1)
            avg_probs = probs.mean(dim=0)

            preds = logits.argmax(dim=1).cpu().tolist()
            w_int = int(weights[i] * 100)
            
            vote_w = max(1, int(w_int / nb_aug))
            for p in preds:
                all_votes.extend([p] * vote_w)

            w_probs = weights[i] * avg_probs
            if total_probs is None: total_probs = w_probs
            else: total_probs += w_probs

    # --- DÉCISION ---
    if total_probs is None: return 0
    max_conf = total_probs.max().item()
    pred_final = int(total_probs.argmax().item())

    if max_conf > 0.65:
        return pred_final
    else:
        return get_majority_vote(all_votes)