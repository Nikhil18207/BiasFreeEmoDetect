# BiasFreeEmoDetect
"BiasCorrectEmoRec" enhances emotion recognition by integrating facial images and physiological signals while mitigating biases in AI models. Using deep learning, it ensures fair and accurate detection across diverse demographics. This project aims to improve reliability and ethical AI practices in emotion analysis.

📌 Project: Bias-Free Emotion Recognition Using Deep Learning
✅ Dataset Setup
Used a facial emotion dataset organized by set_id folders with 8 emotions: Anger, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprised

Merged with metadata (gender, age, country) from emotions.csv

🔍 Bias Analysis
Identified imbalance:

More female samples than male

Overrepresentation from Russia (RU)

Few samples for teen & mid-age groups

Emotion distribution was balanced, but skewed by gender/country

🔁 Bias-Aware Augmentation
Applied targeted augmentations to minority groups (MALE, PH, IN, Teen, Mid-Age)

Techniques: Horizontal flip, brightness, rotation, blur, noise

Dataset size increased:
From 152 → 552 samples for balanced representation

🔄 DataLoader + Preprocessing
Created a custom PyTorch Dataset class

Applied transformations (Resize → Normalize → Augment)

Dataloader working: torch.Size([32, 3, 224, 224]) verified ✅

🧠 Model 1: Custom CNN
Built and trained from scratch

Used class-weighted CrossEntropyLoss

Accuracy after 20 epochs: 99.82%

Final evaluation:

✅ Confusion Matrix = perfect

✅ Classification Report = Precision/Recall/F1 ~ 1.00

✅ AUC-ROC = 1.00 (macro & per-class)

🤖 Model 2: ResNet18 (Pretrained)
Loaded and fine-tuned full ResNet18

Added:

Class weights

Learning rate scheduler

Full fine-tuning (no frozen layers)

Accuracy after 20 epochs: 99.82%

Evaluation step in progress...

🟢 Next:
Evaluate ResNet18 (Confusion Matrix + AUC + Report)

Train Model 3: MobileNetV2

Compare all 3 models side-by-side

Add Grad-CAM & Gender-Wise Fairness Analysis (optional)

Deploy or document for paper/presentation

