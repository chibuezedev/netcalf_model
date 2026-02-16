# Network Intrusion Detection System (IDS) - Deep Learning

## 🎯 Project Overview

A comprehensive, production-ready **Network Intrusion Detection System** using state-of-the-art deep learning architectures for real-time network traffic monitoring and security.

### Key Features

✅ **Multiple Advanced Architectures**:
- Deep MLP with Self-Attention
- 1D CNN for Pattern Recognition
- Bidirectional LSTM with Attention
- Transformer Encoder
- ResNet-inspired Architecture
- Ensemble Model

✅ **Advanced Preprocessing**:
- Multiple scaling methods (Standard, Robust, MinMax)
- Intelligent feature selection (Mutual Information)
- Class balancing (SMOTE, ADASYN, SMOTE-Tomek, etc.)
- PCA support for dimensionality reduction

✅ **Overfitting Prevention**:
- Dropout layers (40-60%)
- Batch normalization
- L2 regularization
- Early stopping
- Learning rate reduction on plateau
- Model checkpointing

✅ **Comprehensive Evaluation**:
- Confusion matrices (raw & normalized)
- ROC curves (all classes)
- Precision-Recall curves
- Classification report heatmaps
- Training history plots
- Prediction confidence analysis
- Per-class metrics

✅ **Production-Ready API**:
- FastAPI RESTful endpoints
- Real-time single prediction
- Batch prediction support
- CORS-enabled for frontend integration
- Health checks and status monitoring

---

## 📊 Dataset: CIC-UNSW-NB15

**Attack Categories** (10 classes):
1. **Benign** - Normal traffic
2. **Analysis** - Traffic analysis attacks
3. **Backdoor** - Unauthorized access attempts
4. **DoS** - Denial of Service
5. **Exploits** - Vulnerability exploitation
6. **Fuzzers** - Fuzzing attacks
7. **Generic** - Cryptographic attacks
8. **Reconnaissance** - Network scanning
9. **Shellcode** - Code injection
10. **Worms** - Self-propagating malware

**Dataset Files**:
- `Data.csv` - 448,915 flows (80% benign, 20% attacks)
- `Label.csv` - Numerical labels (0-9)
- `Readme.txt` - Label mappings

---

## 🏗️ Project Structure

```
network_ids_system/
├── preprocessors/
│   └── advanced_preprocessor.py      # Advanced preprocessing pipeline
├── models/
│   └── architectures.py              # 6 SOTA deep learning architectures
├── utils/
│   └── evaluation.py                 # Comprehensive evaluation suite
├── api/
│   └── api_server.py                 # FastAPI production server
├── train_pipeline.py                 # Complete training pipeline
├── requirements.txt                  # Dependencies
├── README.md                         # This file
└── outputs/                          # Generated outputs
    ├── models/                       # Trained models (.h5)
    ├── visualizations/               # All evaluation plots
    ├── preprocessors/                # Saved preprocessors
    ├── production_model.h5           # Best model
    ├── model_metadata.json           # Model metadata
    └── model_comparison.csv          # Architecture comparison
```

---

## 🚀 Installation

### 1. Clone/Setup Project

```bash
cd ./network_ids_system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

### 3. Prepare Dataset

Place your dataset files in the project root:
- `Data.csv`
- `Label.csv`
- `Readme.txt`

---

## 🎓 Training

### Run Complete Training Pipeline

```bash
python train_pipeline.py
```

This will:
1. Load and analyze the dataset
2. Apply advanced preprocessing with class balancing
3. Train **6 different architectures**:
   - DeepMLP_Attention
   - CNN1D
   - BiLSTM_Attention
   - Transformer
   - ResNet
   - Ensemble
4. Evaluate each model with comprehensive metrics
5. Generate **all visualizations** (50+ plots)
6. Select the best model
7. Save production-ready model

### Training Outputs

**Models** (`outputs/models/`):
- `DeepMLP_Attention_best.h5`
- `CNN1D_best.h5`
- `BiLSTM_Attention_best.h5`
- `Transformer_best.h5`
- `ResNet_best.h5`
- `Ensemble_best.h5`

**Visualizations** (per model):
- Confusion matrix (raw & normalized)
- ROC curves (all classes)
- Precision-Recall curves
- Classification report heatmap
- Class distribution comparison
- Prediction confidence analysis
- Training history (accuracy, loss, precision, recall)

**Metrics** (JSON format):
- Accuracy, Precision, Recall, F1-Score
- Matthews Correlation Coefficient
- Cohen's Kappa
- Per-class metrics

**Production Files**:
- `production_model.h5` - Best performing model
- `model_metadata.json` - Model information
- `preprocessors/preprocessor.pkl` - Fitted preprocessor
- `model_comparison.csv` - All models comparison

---

## 🔧 API Server (FastAPI)

### Start the Server

```bash
cd /home/claude/network_ids_system/api
python api_server.py
```

Server runs on: `http://localhost:8000`

### API Documentation

Interactive docs: `http://localhost:8000/docs`

### Endpoints

#### 1. Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-02-10T10:30:00"
}
```

#### 2. Model Status
```bash
GET /status
```

Response:
```json
{
  "status": "active",
  "model_loaded": true,
  "model_name": "Ensemble",
  "num_classes": 10,
  "class_names": ["Benign", "Analysis", ...],
  "input_features": 50,
  "model_metrics": {
    "accuracy": 0.9876,
    "f1_macro": 0.9654,
    "f1_weighted": 0.9870
  }
}
```

#### 3. Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "features": [0.5, 0.3, ..., 0.8]  # 50 features
}
```

Response:
```json
{
  "success": true,
  "prediction": "DoS",
  "prediction_id": 3,
  "confidence": 0.9876,
  "all_probabilities": {
    "Benign": 0.0023,
    "Analysis": 0.0001,
    "DoS": 0.9876,
    ...
  },
  "action": "BLOCK",
  "timestamp": "2024-02-10T10:30:00",
  "processing_time_ms": 12.5
}
```

#### 4. Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json

{
  "traffic_batch": [
    [0.5, 0.3, ..., 0.8],
    [0.2, 0.7, ..., 0.4],
    ...
  ]
}
```

Response:
```json
{
  "success": true,
  "predictions": [
    {
      "sample_id": 0,
      "prediction": "Benign",
      "prediction_id": 0,
      "confidence": 0.9543,
      "action": "ALLOW",
      "all_probabilities": {...},
      "processing_time_ms": 5.2
    },
    ...
  ],
  "total_samples": 100,
  "processing_time_ms": 520.0,
  "timestamp": "2024-02-10T10:30:00"
}
```

#### 5. Get Classes
```bash
GET /classes
```

---

## Frontend Integration

### JavaScript/Node.js Example

```javascript
async function detectNetworkTraffic(features) {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ features })
  });
  
  const result = await response.json();
  
  if (result.action === 'BLOCK') {
    console.log(`THREAT DETECTED: ${result.prediction}`);
    console.log(`Confidence: ${(result.confidence * 100).toFixed(2)}%`);
  } else {
    console.log('✓ Traffic is benign');
  }
  
  return result;
}

// Batch prediction
async function detectBatch(trafficBatch) {
  const response = await fetch('http://localhost:8000/predict/batch', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ traffic_batch: trafficBatch })
  });
  
  return await response.json();
}
```

### Real-time Streaming

```javascript
const networkStream = getNetworkStream();

networkStream.on('packet', async (packet) => {
  const features = extractFeatures(packet);
  const result = await detectNetworkTraffic(features);
  
  await saveToDatabase({
    timestamp: new Date(),
    features: features,
    prediction: result.prediction,
    action: result.action,
    confidence: result.confidence
  });
  
  if (result.action === 'BLOCK') {
    blockIP(packet.sourceIP);
    sendAlert(result);
  }
});
```

---

## 📈 Research Metrics

All metrics are automatically generated during training:

### Classification Metrics
- **Accuracy** - Overall correctness
- **Precision** (Macro & Weighted) - Positive prediction accuracy
- **Recall** (Macro & Weighted) - True positive detection rate
- **F1-Score** (Macro & Weighted) - Harmonic mean of precision/recall
- **Matthews Correlation Coefficient** - Quality of binary classification
- **Cohen's Kappa** - Inter-rater agreement

### Visual Analytics
- **Confusion Matrices** - Classification errors visualization
- **ROC Curves** - True positive vs false positive rates
- **Precision-Recall Curves** - Precision-recall trade-off
- **Training History** - Loss, accuracy, precision, recall over epochs
- **Class Distribution** - True vs predicted class counts
- **Confidence Analysis** - Prediction confidence distribution

### Per-Class Metrics
- Precision, Recall, F1-Score for each of 10 classes
- Support (number of samples) per class

---

## 🛡️ Overfitting Prevention

### Techniques Implemented

1. **Dropout** (40-60%) - Randomly drop neurons during training
2. **Batch Normalization** - Normalize layer inputs
3. **L2 Regularization** (λ=0.001) - Penalize large weights
4. **Early Stopping** - Stop when validation loss stops improving
5. **Learning Rate Reduction** - Reduce LR on plateau
6. **Data Augmentation** - SMOTE-based oversampling
7. **Cross-Validation** - Train/Validation/Test split
8. **Gradient Clipping** - Prevent exploding gradients

---

## 🎯 Model Performance

Expected performance on CIC-UNSW-NB15:

| Metric | Target | Expected |
|--------|--------|----------|
| Accuracy | >95% | 96-98% |
| F1-Score (Weighted) | >95% | 96-98% |
| Precision (Macro) | >90% | 92-95% |
| Recall (Macro) | >90% | 92-95% |
| False Positive Rate | <5% | 2-4% |

---

## 🔍 Model Architectures Details

### 1. Deep MLP with Attention
- 4 dense blocks (256→128→64→32)
- Self-attention mechanism
- Best for: General-purpose detection

### 2. 1D CNN
- 3 convolutional blocks
- Pattern recognition in feature sequences
- Best for: Spatial pattern detection

### 3. Bidirectional LSTM
- 2 BiLSTM layers (128→64)
- Attention pooling
- Best for: Temporal dependencies

### 4. Transformer
- Multi-head attention (4 heads)
- Feed-forward network
- Best for: Complex relationships

### 5. ResNet
- Skip connections
- 2 residual blocks
- Best for: Deep feature learning

### 6. Ensemble
- Combines MLP, CNN, and Attention
- Robust predictions
- Best for: Overall performance

---

## 📝 Citation

If you use this system in your research, please cite:

```
Mohammadian, H., Lashkari, A. H., & Ghorbani, A. (2024).
Poisoning and Evasion: Deep Learning-Based NIDS under Adversarial Attacks.
21st Annual International Conference on Privacy, Security and Trust (PST).
```

---

## 🤝 Integration with Node.js Backend

The FastAPI server is designed to work alongside your Node.js main server:

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   Frontend      │ ───────▶│   Node.js       │ ───────▶│   FastAPI       │
│   (React/Vue)   │         │   Main Server   │         │   ML Server     │
│                 │◀─────── │   (Port 3000)   │◀─────── │   (Port 8000)   │
└─────────────────┘         └─────────────────┘         └─────────────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │   Database      │
                            │   (MongoDB/     │
                            │    PostgreSQL)  │
                            └─────────────────┘
```

---
**Start training now:**
```bash
python train_pipeline.py
```

**Start API server:**
```bash
cd api && python api_server.py
```
