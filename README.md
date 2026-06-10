# Face Recognition - Ensemble Learning Pipeline

This repository contains the source code for a robust face recognition system developed during my AI Project in London (December 2025). 

**Performance:** Achieved **80.65% accuracy**, ranking **12th out of 119 AI engineers**.

## 🚀 Architecture Overview
The system uses an **Ensemble Learning** strategy combining 4 distinct deep learning architectures to maximize prediction diversity and robustness against variations in clothing, lighting, and cluttered backgrounds:
1. **ConvNeXt Small (Global)** - Trained with heavy geometric augmentations (45° rotations).
2. **ResNet-50 & ResNet-101 (Snipers)** - Fine-tuned on facial features with targeted deep layer unfreezing.
3. **DenseNet-121 (Inverted)** - Integrated for perspective diversity.

## 🛠️ Key Features
- **Heavy Test-Time Augmentation (TTA):** Multi-view generation per image (aspect-ratio resizing, center crops, and gamma adjustments).
- **Hybrid Voting System:** Probability fusion (soft-voting) paired with a majority vote fallback (hard-voting) under a 65% confidence threshold.
