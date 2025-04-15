# 🎭 Bias-Free Emotion Recognition using Deep Learning

A deep learning-based system to classify facial emotions fairly across gender, age, and region while maintaining high performance and interpretability.

---

## 📂 Dataset

**Structure:**

- Folder: `images/set_id/*.jpg`
- Each `set_id` contains 8 emotion images:
  - Anger, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprised

**Metadata:**

- CSV file: `emotions.csv`
- Columns: `set_id`, `gender`, `age`, `country`

---

## 🔍 Bias Analysis

We identified major biases in the dataset:

- **Gender imbalance**: Majority of samples are female
- **Country bias**: Overrepresentation from Russia (RU)
- **Age gaps**: Sparse data for Teen, Mid-Age, and Senior groups
- **Emotion classes**: Balanced overall, but skewed across demographics

---

## 🔁 Bias-Aware Data Augmentation

To ensure fairness, we applied targeted augmentations to:

- **👤 Gender**: Male samples
- **🌍 Countries**: India (IN), Philippines (PH)
- **🎂 Age groups**: Teen, Mid-Age, Senior

**Techniques used:**

- Horizontal Flip
- Rotation
- Brightness & Contrast adjustment
- Gaussian Noise
- Motion Blur

**➡️ Final Dataset Size**: 152 → 552 samples

---

## ⚙️ Preprocessing & Dataloader

- Images resized to `224x224`
- Normalized using ImageNet stats
- PyTorch `Dataset` class created
- Real-time augmentations applied during training
- Dataloader verified with shape: `torch.Size([32, 3, 224, 224])`

---

## 🧠 Model 1: Custom CNN

**Architecture:**

- 3 × Conv → BatchNorm → ReLU → MaxPool
- Fully Connected + Dropout
- Output: 8 emotion classes

**Training Setup:**

- Weighted CrossEntropy Loss
- Adam Optimizer
- 20 Epochs

**Results:**

- ✅ **Accuracy**: 99.82%
- 📊 **Precision / Recall / F1-score**: ~1.00
- 📈 **AUC-ROC (macro)**: 1.00
- ✅ Perfect Confusion Matrix

---

## 🤖 Model 2: ResNet18 (Pretrained)

**Modifications:**

- Replaced final FC layer with 8-output layer
- Fine-tuned all layers (no freezing)
- Used weighted loss + learning rate scheduler

**Training:**

- 20 Epochs
- Adam Optimizer
- Class Weighting + StepLR Scheduler

**Results:**

- ✅ **Accuracy**: 99.82%
- 📈 Final evaluation in progress...

---

## 🚀 Next Steps

- ✅ Final evaluation (Confusion Matrix + AUC + Report) for ResNet18
- 🔜 Train Model 3: MobileNetV2 (lightweight & fast)
- 📊 Compare all 3 models side-by-side
- 🎨 Add Grad-CAM for visual interpretability
- ⚖️ Analyze fairness across gender (accuracy parity)
- 📦 Optional deployment via Streamlit or Gradio

---

## 🎯 Goal

To develop a robust, fair, and explainable facial emotion recognition model that performs equally well across all demographic groups.
