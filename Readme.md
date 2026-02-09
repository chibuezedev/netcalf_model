# Vision Transformer for Skin Disease Classification

Classification of skin diseases using Vision Transformer (ViT) with explainable AI capabilities.

## Dataset
Five classes: Acne, Eczema, Herpes, Panu, Rosacea

## Features
- Custom ViT implementation from scratch
- Attention-based explainability (XAI heatmaps)
- Data augmentation
- Automated model checkpointing
- Performance visualization

## Requirements
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Dataset Structure
```
train/
├── acne/
├── eczema/
├── herpes/
├── panu/
└── rosacea/
```

## Usage
```bash
python main.py
```

## Architecture
- **Patch Size**: 16×16
- **Embedding Dim**: 768
- **Depth**: 12 transformer blocks
- **Heads**: 12 attention heads
- **Parameters**: ~86M

## Training Configuration
- Epochs: 10 # (use 20 in better gpu)
- Batch Size: 16 (use 32 in better gpu)
- Optimizer: AdamW (lr=3e-4, weight_decay=0.05)
- Scheduler: Cosine Annealing
- Train/Val Split: 80/20

## Outputs
- `vit_best_model.pth` - Best model weights
- `confusion_matrix.png` - Classification performance
- `training_history.png` - Loss and accuracy curves
- `attention_visualization.png` - XAI attention maps

## Performance Optimization
For faster training:
- Reduce batch size: `batch_size=8`
- Smaller model: `depth=6, embed_dim=384, num_heads=6`
- Lower resolution: `img_size=128`

## License
MIT