# Baseline Industrial Defect Classifier: Layer-by-Layer Architecture

**Implementation Module:** [`src/baseline_cnn.py`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/src/baseline_cnn.py)  
**Total Trainable Parameters:** 28,358 parameters  
**Architecture Diagram:** [`docs/figures/baseline_cnn_architecture.png`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/figures/baseline_cnn_architecture.png)

---

## 1. High-Level Architecture Flow

```
Input Surface Patch [B, 3, 200, 200]
         ↓
  [Stage 1: Low-Level Feature Extractor]
  • Conv2d (3 -> 16, kernel 3x3, pad 1, stride 1)
  • BatchNorm2d (16) + ReLU()
  • MaxPool2d (kernel 2x2, stride 2)
  Output: [B, 16, 100, 100]
         ↓
  [Stage 2: Mid-Level Texture Extractor]
  • Conv2d (16 -> 32, kernel 3x3, pad 1, stride 1)
  • BatchNorm2d (32) + ReLU()
  • MaxPool2d (kernel 2x2, stride 2)
  Output: [B, 32, 50, 50]
         ↓
  [Stage 3: High-Level Morphology Extractor]
  • Conv2d (32 -> 64, kernel 3x3, pad 1, stride 1)
  • BatchNorm2d (64) + ReLU()
  • MaxPool2d (kernel 2x2, stride 2)
  Output: [B, 64, 25, 25]
         ↓
  [Stage 4: Global Aggregation & Classifier Head]
  • AdaptiveAvgPool2d(1, 1)  ──► [B, 64, 1, 1]
  • Flatten()                ──► [B, 64]
  • Linear(64 -> 64) + ReLU()
  • Dropout(p = 0.25)
  • Linear(64 -> 6)          ──► [B, 6] (6 Defect Logits)
```

---

## 2. Layer-by-Layer Specification Table

| Stage | Layer Name | Layer Type | Kernel Size | Stride | Pad | In Channels / Dims | Out Channels / Dims | Output Tensor Shape (Batch $B=32$) | Parameter Count |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Input** | `input` | Tensor | — | — | — | — | — | `[32, 3, 200, 200]` | $0$ |
| **Stage 1** | `conv1` | `nn.Conv2d` | $3 \times 3$ | 1 | 1 | 3 | 16 | `[32, 16, 200, 200]` | $3 \times 16 \times 3 \times 3 + 16 = 448$ |
| | `bn1` | `nn.BatchNorm2d` | — | — | — | 16 | 16 | `[32, 16, 200, 200]` | $32$ |
| | `act1` | `nn.ReLU` | — | — | — | 16 | 16 | `[32, 16, 200, 200]` | $0$ |
| | `pool1` | `nn.MaxPool2d` | $2 \times 2$ | 2 | 0 | 16 | 16 | `[32, 16, 100, 100]` | $0$ |
| **Stage 2** | `conv2` | `nn.Conv2d` | $3 \times 3$ | 1 | 1 | 16 | 32 | `[32, 32, 100, 100]` | $16 \times 32 \times 3 \times 3 + 32 = 4,640$ |
| | `bn2` | `nn.BatchNorm2d` | — | — | — | 32 | 32 | `[32, 32, 100, 100]` | $64$ |
| | `act2` | `nn.ReLU` | — | — | — | 32 | 32 | `[32, 32, 100, 100]` | $0$ |
| | `pool2` | `nn.MaxPool2d` | $2 \times 2$ | 2 | 0 | 32 | 32 | `[32, 32, 50, 50]` | $0$ |
| **Stage 3** | `conv3` | `nn.Conv2d` | $3 \times 3$ | 1 | 1 | 32 | 64 | `[32, 64, 50, 50]` | $32 \times 64 \times 3 \times 3 + 64 = 18,496$ |
| | `bn3` | `nn.BatchNorm2d` | — | — | — | 64 | 64 | `[32, 64, 50, 50]` | $128$ |
| | `act3` | `nn.ReLU` | — | — | — | 64 | 64 | `[32, 64, 50, 50]` | $0$ |
| | `pool3` | `nn.MaxPool2d` | $2 \times 2$ | 2 | 0 | 64 | 64 | `[32, 64, 25, 25]` | $0$ |
| **Stage 4** | `gap` | `nn.AdaptiveAvgPool2d`| — | — | — | 64 | 64 | `[32, 64, 1, 1]` | $0$ |
| | `flat` | `nn.Flatten` | — | — | — | $64 \times 1 \times 1$ | 64 | `[32, 64]` | $0$ |
| | `fc1` | `nn.Linear` | — | — | — | 64 | 64 | `[32, 64]` | $64 \times 64 + 64 = 4,160$ |
| | `act4` | `nn.ReLU` | — | — | — | 64 | 64 | `[32, 64]` | $0$ |
| | `drop` | `nn.Dropout(0.25)` | — | — | — | 64 | 64 | `[32, 64]` | $0$ |
| | `fc2` | `nn.Linear` | — | — | — | 64 | 6 | `[32, 6]` (Logits) | $64 \times 6 + 6 = 390$ |
| **Total** | — | — | — | — | — | — | — | — | **28,358 Trainable Parameters** |

---

## 3. Layer-Level Architectural Justification

1. **Why $3 \times 3$ Small Convolutions?**  
   Following the Simonyan & Zisserman (VGG) principle, stacking small $3 \times 3$ convolutional filters achieves the same effective receptive field as larger $5 \times 5$ filters ($2 \text{ layers} \to 5 \times 5$) but uses significantly fewer parameters and introduces additional non-linear ReLU activations.
2. **Why Batch Normalization (`BatchNorm2d`)?**  
   Standardizes intermediate feature distributions ($\mu=0, \sigma^2=1$) across the batch, reducing internal covariate shift, smoothing the optimization landscape, and accelerating training convergence.
3. **Why Global Average Pooling (GAP) instead of Flattening $25 \times 25$?**  
   If we flattened $64 \times 25 \times 25 = 40,000$ features directly into a dense layer, a single dense layer would require $40,000 \times 64 = 2.56 \text{ million}$ parameters! This would cause severe overfitting on our 1,260 training images. GAP reduces the spatial dimensions to $1 \times 1$, constraining the entire network to just **28,358 parameters**, ensuring fast, lightweight training on student hardware with zero risk of GPU/CPU out-of-memory errors.
