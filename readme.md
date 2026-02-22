# 🍽️ Inverse Cooking: Recipe Generation from Food Images
> A deep learning project that generates cooking recipes from food images using CNN and Transformer architecture.

---

## 📖 Overview

This project implements an **image-to-text generation model** that can look at a food image and generate step-by-step cooking instructions. Built with PyTorch, it combines a CNN encoder (ResNet18) for visual feature extraction and a Transformer decoder for sequential text generation.

### Key Features
- 🖼️ **Image Understanding**: Uses pre-trained ResNet18 to extract visual features
- 📝 **Recipe Generation**: Transformer decoder generates coherent cooking instructions
- 🎯 **End-to-End Training**: Single pipeline from image input to text output
- 📊 **13,463 Recipes**: Trained on diverse food dataset with real recipes
- ⚡ **GPU Optimized**: Efficient training with CUDA support

---

## 🎯 Problem Statement

**Challenge**: Given an image of a finished dish, automatically generate the cooking recipe.

**Why It Matters**:
- Helps home cooks recreate dishes they see
- Assists people with dietary restrictions in understanding ingredients
- Educational tool for culinary students
- Accessibility feature for recipe discovery

---

## 🗃️ Architecture

### Model Design

```
Input Image (128×128×3)
         ↓
    CNN Encoder (ResNet18)
         ↓
    Feature Vector (256)
         ↓
  Transformer Decoder (2 layers, 4 heads)
         ↓
    Recipe Text Output
```

### Components

1. **CNN Encoder**
   - Pre-trained ResNet18 (ImageNet weights)
   - Fine-tuned last layer
   - Output: 256-dimensional feature vector

2. **Transformer Decoder**
   - 2 decoder layers
   - 4 attention heads
   - Embedding dimension: 256
   - Vocabulary size: 8,215 words
   - Max sequence length: 512 tokens

3. **Training Strategy**
   - Teacher forcing during training
   - Auto-regressive generation during inference
   - Cross-entropy loss
   - AdamW optimizer with learning rate scheduling

---

## 📊 Dataset

- **Source**: [Food Ingredients and Recipe Dataset](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images)
- **Total Recipes**: 13,463
- **Train/Val Split**: 80/20
- **Image Size**: 128×128 RGB
- **Data Augmentation**: Random crop, horizontal flip, color jitter

---

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
CUDA 11.0+ (for GPU support)
```

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/WALKMAN303/inverse-cooking.git
cd inverse-cooking
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download pre-trained models** (See [Model Files](#-model-files) section below)
```bash
python download_models.py
```

4. **Run the script**
```bash
python app.py
```

That's it! You're ready to generate recipes from food images! 🎉

---

## 📦 Model Files

> **Note**: Due to GitHub's file size limitations (100 MB max), the pre-trained model weights are hosted on Google Drive.

### Download Pre-trained Models

**Option 1: Automatic Download (Recommended)**
```bash
python download_models.py
```

**Option 2: Manual Download**

Download the following files and place them in the project root directory:

| File | Size | Description | Download Link |
|------|------|-------------|---------------|
| `best_model.pth` | 118 MB | Best checkpoint model | [Download](https://drive.google.com/file/d/1HfSb-zVBlxTf22YrheEVGbT7KjZHoe-C/view?usp=sharing) |
| `vocab.pkl` | <1 MB | Vocabulary object | [Download](https://drive.google.com/file/d/13S6BM-Fc5uPCO-Tn1VJxxUnx3U0zFlHD/view?usp=sharing) |

**After downloading, your project structure should look like:**
```
inverse-cooking/
├── best_model.pth                ✅
├── vocab.pkl                     ✅
├── inverse-cooking.py
├── download_models.py
├── requirements.txt
├── README.md
└── image
```

### Alternative: Train Your Own Models

If you prefer to train from scratch instead of using pre-trained weights:
```bash

jupyter notebook inverse-cooking.ipynb

python train.py --epochs 15 --batch-size 32 --lr 3e-4
```

Training takes approximately 1 hour on Tesla T4 GPU and use kaggle or colab for the T4 GPU.

---

## 💻 Usage

### Quick Inference

```python
from model import InverseCookingModel, generate_recipe
import torch
from PIL import Image

# Load trained model
model = InverseCookingModel(embed_size=256, vocab_size=8215, ...)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Generate recipe from image
image_path = 'test_images/pizza.jpg'
recipe = generate_recipe(image_path, model, vocab, transform, device)
print(recipe)
```

---

## 📈 Results

### Training Performance

| Metric | Value |
|--------|-------|
| Best Validation Loss | 2.96 |
| Training Time | ~1 hours (15 epochs on Tesla T4) |
| Final Train Loss | 2.92 |
| Model Parameters | 17.7M |

### Sample Predictions

**Example: Pizza**
- **Generated**: "preheat oven to 450 degrees, roll out pizza dough spread tomato sauce add mozzarella cheese top with pepperoni bake for 15 minutes until cheese is melted and bubbly."
- **Quality**: Coherent, logical sequence ✅

---

## 🛠️ Technical Details

### Hyperparameters

```python
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS = 15
EMBED_SIZE = 256
NUM_HEADS = 4
NUM_LAYERS = 2
MAX_SEQ_LENGTH = 512
```

### Model Architecture Details

```
CNNEncoder(
  (resnet): Sequential(...)
  (linear): Linear(512 → 256)
  (bn): BatchNorm1d(256)
  (relu): ReLU()
)

TransformerDecoder(
  (embedding): Embedding(8215, 256)
  (positional_encoding): Parameter(512, 256)
  (transformer_decoder): TransformerDecoder(2 layers)
  (fc_out): Linear(256 → 8215)
)

Total Parameters: 17,761,623
Trainable Parameters: 7,042,815
```

---

## 📁 Project Structure

```
inverse-cooking/
├── inverse-cooking.py      # Main training notebook
├── model.py                   # Model architecture
├── download_models.py         # Script to download pre-trained models
├── requirements.txt           # Python dependencies
├── best_model.pth             # Trained model weights (download required)
├── vocab.pkl                  # Vocabulary object (download required)
├── README.md                  # This file
└── images/                    # Sample images and results
    ├── architecture.png
    └── training_history.png
```

---

## 🔧 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
pandas>=1.5.0
pillow>=9.0.0
scikit-learn>=1.2.0
matplotlib>=3.5.0
kagglehub>=0.1.0
gdown>=4.7.1
```

---

## 🎓 Learning Outcomes

This project demonstrates:
- **Deep Learning**: CNN + Transformer architecture
- **Computer Vision**: Image feature extraction with ResNet
- **NLP**: Sequential text generation with attention
- **PyTorch**: Custom datasets, data loaders, training loops
- **Best Practices**: Modular code, documentation, version control
- **MLOps**: Model versioning and external storage for large files

---

## 🚧 Future Improvements

- [ ] Add ingredient extraction as a separate task
- [ ] Implement beam search for better generation
- [ ] Add BLEU/ROUGE metrics for evaluation
- [ ] Fine-tune with larger models (ResNet50, ViT)
- [ ] Create web demo with Gradio/Streamlit
- [ ] Add nutritional information prediction

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
---

## 👤 Author

**Your Name**
- GitHub: [@WALKMAN303](https://github.com/WALKMAN303)
- LinkedIn: [Arjun K R](https://linkedin.com/in/arjun-k-r-)
---

## 📞 Contact

Have questions or suggestions? Feel free to:
- Open an issue
- Reach out on LinkedIn
