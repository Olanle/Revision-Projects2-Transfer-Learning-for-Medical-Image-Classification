# Chest X-Ray Pneumonia Detection (Transfer Learning)

Fine-tuning ResNet50 on chest X-ray images to classify patients as NORMAL or PNEUMONIA using transfer learning in PyTorch.

---

## Problem Statement

Pneumonia is a life-threatening condition where early detection is critical. This project builds a binary image classifier using a pretrained ResNet50 model fine-tuned on chest X-ray images. The core challenge beyond model accuracy is class imbalance — the dataset contains nearly 3x more pneumonia cases than normal cases, which directly impacts how reliably the model detects each class.

---

## Dataset

Chest X-Ray Pneumonia — sourced from Kaggle (paultimothymooney/chest-xray-pneumonia)

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,341 | 3,875 | 5,216 |
| Val | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

Class imbalance ratio: approximately 1:3 (NORMAL:PNEUMONIA). The validation set of only 16 images produced unstable accuracy readings during training. All final evaluation was done on the 624-image test set.

---

## Transfer Learning Approach

ResNet50 was pretrained on ImageNet. Its learned visual features — edges, textures, gradients — transfer well to medical imaging even though it has never seen a chest X-ray. With only 1,341 normal training images, building from scratch would cause severe overfitting.

**Modifications made:**
- Loaded ResNet50 with pretrained ImageNet weights
- Froze all convolutional layers (`requires_grad = False`)
- Replaced final layer: `Linear(2048 → 1000)` with `Linear(2048 → 2)`
- Only trained the new final layer: `optim.Adam(model.fc.parameters())`

---

## Training Setup

| Component | Value |
|-----------|-------|
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Trainable Parameters | Final layer only |
| Epochs | 10 |
| Batch Size | 32 |
| Device | CUDA (GPU required — CPU was ~20min/epoch) |

---

## Results

**Test Accuracy: 80.9%** (624 images)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| NORMAL | 0.94 | 0.53 | 0.67 | 234 |
| PNEUMONIA | 0.77 | 0.98 | 0.87 | 390 |
| Overall | 0.84 | 0.81 | 0.79 | 624 |

---

## The Recall Problem and Its Effect on Detection Accuracy

The most important finding in this project is not the overall accuracy — it is the gap between the two recall scores.

**PNEUMONIA recall: 0.98** — the model catches 98% of all sick patients. Almost no pneumonia case is missed.

**NORMAL recall: 0.53** — the model only correctly identifies 53% of healthy patients. Nearly half of all healthy patients are incorrectly flagged as having pneumonia.

This gap is a direct consequence of class imbalance. During training the model saw 3,875 pneumonia images and only 1,341 normal images. It learned that pneumonia is the more frequent outcome and defaulted toward predicting it. The result is a model that is highly sensitive to pneumonia but poor at confirming health.

**Why this matters in practice:** In a real clinical deployment, a model with 0.53 normal recall means roughly 1 in 2 healthy patients gets incorrectly referred for further testing — creating unnecessary anxiety, cost, and strain on medical resources. The 80.9% overall accuracy number hides this problem entirely, which is why recall per class must always be reported in medical ML, not just overall accuracy.

**How to fix it:**
- WeightedRandomSampler — oversample NORMAL during training so the model sees equal class frequencies
- Weighted loss — penalise NORMAL misclassifications more heavily during training
- Collect more normal X-ray images to reduce the imbalance naturally

---

## Single Image Inference

The model was tested on individual uploaded X-rays with confidence scores. A pneumonia case was correctly identified at **90.22% confidence**.

---

## Key Learnings

- Transfer learning prevents overfitting on small medical datasets
- Only pass `model.fc.parameters()` to the optimizer when layers are frozen
- Class imbalance directly causes biased recall — always check class distribution before training
- Overall accuracy is a misleading metric in imbalanced medical datasets — always report per-class recall
- GPU is essential for fine-tuning large models like ResNet50

---

## Stack
`PyTorch` `torchvision` `scikit-learn` `PIL` `Kaggle`

---

*Part of a 10-project ML engineering curriculum targeting Edge/Embedded ML roles.*
