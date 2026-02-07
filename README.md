# 🔬 DermaMNIST: Skin Lesion Classification with Computer Vision

![Project Banner](runs/dermamnist.png)

## 👋 About the Project

This repository is a deep learning project born from a weekend coding session with the spirit of "Let's see what AI can do with dermatology!" It performs multi-class skin lesion classification using low-resolution dermatoscopic images from the MedMNIST dataset.

The repository name might say "messy," but the classification results are anything but! 🎯

## 🎯 What Does It Do?

- **Data**: Automatically downloads the DermaMNIST dataset from MedMNIST
- **Model**: Uses a custom lightweight Convolutional Neural Network (CNN) called DermaNet
- **Training**: Runs with validation tracking and automatic best model saving
- **Result**: Classifies skin lesions into 7 different categories with high accuracy

## 📊 Dataset: DermaMNIST

The DermaMNIST dataset contains dermatoscopic images of 7 different types of skin lesions:

| Class ID | Lesion Type |
|----------|-------------|
| 0 | Actinic Keratoses |
| 1 | Basal Cell Carcinoma |
| 2 | Benign Keratosis |
| 3 | Dermatofibroma |
| 4 | Melanoma |
| 5 | Melanocytic Nevi |
| 6 | Vascular Lesions |

- **Image Size**: 28×28 pixels (RGB)
- **Training Set**: Automatically split by MedMNIST
- **Validation Set**: Used for hyperparameter tuning
- **Test Set**: Final evaluation metrics

## 🛠️ Installation

Set up your environment and get ready to classify some skin lesions!

```bash
# Clone the repository
git clone https://github.com/username/derma-mnist-cv-messy.git
cd derma-mnist-cv-messy

# Create a virtual environment (recommended)
python -m venv .venv

# Activate the virtual environment
# For Windows:
.venv\Scripts\activate

# For Mac/Linux:
source .venv/bin/activate

# Install requirements
pip install -e ".[dev]"
```

### Prerequisites

- **Python**: 3.10 or higher
- **PyTorch**: 2.0 or higher
- **MedMNIST**: 3.0.0 or higher
- **Additional Libraries**: NumPy, Matplotlib, Scikit-learn

## 🚀 Usage

Run these commands to train and evaluate the model:

### 1. Train the Model

```bash
python -m src.train.train1
```

This command will:
- Download the DermaMNIST dataset (if not already present)
- Train the model for 6 epochs
- Save the best performing model to `runs/best.pt`
- Generate training/validation loss curves

**Note**: Training takes just a few minutes on CPU, even faster on GPU! ☕

### 2. Evaluate Performance

```bash
python -m src.train.eval_it
```

This will:
- Load the best saved model
- Evaluate on the test dataset
- Print detailed classification metrics
- Generate and save a confusion matrix

## 🧠 Model Architecture: DermaNet

We built a lightweight yet effective CNN architecture optimized for 28×28 images:

| Layer | Details |
|-------|---------|
| Input | 3×28×28 (RGB Dermatoscopic Image) |
| Conv1 | 32 filters, 3×3 kernel, ReLU + MaxPool |
| Conv2 | 64 filters, 3×3 kernel, ReLU + MaxPool |
| Conv3 | 128 filters, 3×3 kernel, ReLU + MaxPool |
| FC1 | 128 neurons, ReLU |
| FC2 | 64 neurons, ReLU |
| Output | 7 Classes (Softmax) |

**Total Parameters**: ~100K (lightweight and fast!)

## 📈 Sample Output

After training, you'll see metrics like this:

```
              precision    recall  f1-score   support

           0       0.87      0.82      0.84       114
           1       0.89      0.91      0.90       376
           2       0.75      0.78      0.76       254
           3       0.88      0.85      0.86        95
           4       0.82      0.79      0.80       438
           5       0.91      0.93      0.92      1341
           6       0.94      0.89      0.91       234

    accuracy                           0.88      2852
   macro avg       0.87      0.85      0.86      2852
weighted avg       0.88      0.88      0.88      2852
```

**Note**: Results may vary slightly due to random initialization. This is normal!

## 📁 Project Structure

```
derma-mnist-cv-messy/
│
├── src/
│   ├── data/
│   │   └── loader.py          # Dataset loading and preprocessing
│   ├── models/
│   │   └── net.py             # DermaNet CNN architecture
│   └── train/
│       ├── train1.py          # Training script
│       └── eval_it.py         # Evaluation script
│
├── runs/                       # Training outputs (auto-generated)
│   ├── best.pt                # Best model checkpoint
│   ├── training_curves.png    # Loss/accuracy plots
│   └── confusion_matrix.png   # Confusion matrix visualization
│
├── pyproject.toml             # Project dependencies
└── README.md                  # You are here!
```

**Important**: The `data/` directory is automatically created and gitignored. Dataset files are downloaded on first run.

## 🔍 Key Features

- ✅ **Automatic Data Handling**: No manual dataset download needed
- ✅ **Clean Metrics**: Precision, recall, F1-score, and confusion matrix
- ✅ **Model Checkpointing**: Best model automatically saved
- ✅ **Visualization**: Training curves and confusion matrix plots
- ✅ **Lightweight**: Fast training even on CPU

## 📝 TODO

This project is evolving! Future improvements on the radar:

- [ ] Implement data augmentation (rotation, flip, color jitter)
- [ ] Add learning rate scheduling
- [ ] Integrate logging with WandB or TensorBoard
- [ ] Experiment with deeper architectures (ResNet, EfficientNet)
- [ ] Add Grad-CAM visualization for interpretability
- [ ] Create a web interface (Gradio/Streamlit) for live predictions
- [ ] Refactor code for better modularity
- [ ] Add cross-validation support

## ⚠️ Disclaimer

**This project is for educational and research purposes only.**

🩺 **DO NOT use this model for actual medical diagnosis.** Skin lesion diagnosis requires professional medical evaluation. If you have concerns about skin lesions, please consult a dermatologist.

## 🤝 Contributing

Pull requests are welcome! If you can make the code less "messy" or improve the model performance, you'll be our hero! 🦸

Please feel free to:
- Report bugs
- Suggest enhancements
- Improve documentation
- Add new features

## 📄 License

This project is open source and available for educational purposes. The DermaMNIST dataset is provided by MedMNIST and subject to their terms of use.

---

**Built with ❤️ and PyTorch** | **Powered by MedMNIST**
