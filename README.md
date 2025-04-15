🎭 Bias-Free Emotion Recognition using Deep Learning
This project aims to build an accurate, fair, and explainable emotion recognition model using facial images. The focus is on addressing demographic bias (gender, age, country) while maintaining high classification performance.

📂 Dataset
Facial Emotion Dataset (folder format):
images/set_id/*.jpg

Each set_id contains 8 facial emotion images: Anger, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprised

Metadata CSV:
emotions.csv containing set_id, gender, age, and country

🧪 Bias Analysis
Gender Imbalance: More FEMALE samples than MALE

Country Bias: Overrepresentation from Russia (RU)

Age Gaps: Few samples from Teen, Mid-Age, and Senior age groups

Emotion classes were balanced, but demographically skewed

🔁 Bias-Aware Data Augmentation
To address the imbalance, targeted augmentations were applied to:

MALE faces

Underrepresented countries: IN, PH

Age groups: Teen, Mid-Age, Senior

Augmentation techniques used:

Horizontal flip

Brightness/contrast adjustment

Rotation

Blur

Gaussian noise

➡️ Dataset expanded from 152 → 552 samples

⚙️ Preprocessing & DataLoader
Images resized to 224x224

Normalized using ImageNet mean/std

PyTorch Dataset class with on-the-fly augmentations

Verified with DataLoader: torch.Size([32, 3, 224, 224])

🧠 Model 1: Custom CNN (from Scratch)
Built a simple CNN with:

3 Convolution layers

BatchNorm, MaxPooling

2 Fully Connected layers

Dropout for regularization

Trained with:

Weighted CrossEntropyLoss

Adam Optimizer

20 Epochs

🎯 Final Accuracy: 99.82%
📈 Evaluation:

Precision/Recall/F1: ~1.00

AUC-ROC (Macro): 1.00

Confusion Matrix: Perfect

🤖 Model 2: ResNet18 (Pretrained)
Loaded ResNet18 via torchvision.models

Modified final layer for 8 emotion classes

Fine-tuned all layers (no freezing)

Added:

Class weights for fairness

Learning rate scheduler (StepLR)

🧠 Trained for 20 Epochs
🎯 Final Accuracy: 99.82%

📈 Evaluation in progress...

🚀 What’s Next
✅ Evaluate ResNet18 (Confusion Matrix, AUC, Report)

🔜 Train Model 3: MobileNetV2 (lightweight & fast)

📊 Compare all 3 models side-by-side

🎨 Add Grad-CAM visualizations for interpretability

⚖️ Analyze fairness across gender (accuracy parity)

📦 Optional: Deploy using Streamlit or Gradio

💡 Goal
To build a robust, fair, and explainable deep learning model for facial emotion recognition that works equally well across genders, ages, and regions.

