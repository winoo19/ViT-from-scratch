## Vision Transformer (ViT) From Scratch

A comprehensive PyTorch implementation of the Vision Transformer (ViT) architecture, designed for image classification tasks. This project is built from the ground up, making it an excellent resource for understanding the inner workings of ViT. By implementing key components such as patch embedding, multi-head self-attention, and layer normalization using only basic tensor operations and linear layers, this repository provides a hands-on approach to learning modern transformer-based architectures. It is suitable for both beginners looking to grasp the fundamentals and advanced users aiming to customize and experiment with the ViT model for their own datasets and tasks.

### Features

- Builds ViT blocks (patch embedding, multi‑head self‑attention, MLP) from first principles  
- Supports custom image datasets and transformations  
- Configurable hyperparameters: number of patches, hidden size, heads, layers, dropout  
- Training loop with logging, checkpointing, and basic augmentation  
- Evaluation metrics: accuracy, loss curves, confusion matrix

### Requirements

- Python 3.10+  
- PyTorch 1.12+  
- torchvision  
- numpy  
- matplotlib  
- tqdm

### Installation

```bash
git clone https://github.com/yourusername/ViT-from-scratch.git
cd ViT-from-scratch
pip install -r requirements.txt
```

### Project Structure

```
.
├── data/                     # Downloaded or preprocessed datasets, should be separated into train/ and val/ folders
├── models/                   # Vision Transformer trained models
│   ├── model_1.pt           # Example model checkpoint
│   ├── model_2.pt           # Example model checkpoint
│   └── model_3.pt           # Example model checkpoint
├── src/
│   ├── train.py                 # Training and validation loops
│   └── evaluate.py              # Model evaluation and plotting
│   ├── utils.py                 # Utility functions
│   ├── models.py                # Vision Transformer model definition
│   ├── data.py              # Custom dataset class for loading images
├── tests/
│   ├── test_models.py             # Tests for model components
├── metrics.csv             # model performance metrics
├── .gitignore
├── requirements.txt
└── README.md
```

### Usage

1. Prepare your dataset under `data/` (e.g., CIFAR‑10, custom folders).  
2. Adjust hyperparameters in `src/train.py`.  
3. Launch training:
  ```bash
    python -m src.train
  ```

### Evaluation

After training, adjust model name in evaluate and run:

```bash
  python -m src.evaluate
```

This will output the classification accuracy.

### Acknowledgments

- “An Image is Worth 16×16 Words” by Dosovitskiy et al.

