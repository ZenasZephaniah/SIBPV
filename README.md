# Satellite Imagery-Based Property Valuation 🏠🛰️

## Overview
This project is a **Multimodal Regression Pipeline** that predicts property market values by combining **tabular data** (traditional housing features) with **satellite imagery** (visual environmental context). By integrating "curb appeal" and neighborhood characteristics (green cover, road density) into the pricing model, we achieve higher accuracy than traditional data-only models.

## Objective
- **Predict:** Property value (Target: `price`).
- **Input:** Hybrid dataset of Tabular Features (bedrooms, sqft) + Satellite Images (fetched via Lat/Long).
- **Model:** Late-Fusion Architecture (ResNet18 CNN + MLP).
- **Explainability:** Grad-CAM visualizations to interpret visual value drivers.

## Tech Stack
- **Deep Learning:** PyTorch, torchvision (ResNet18)
- **Data Handling:** Pandas, NumPy
- **Image Processing:** OpenCV, PIL
- **Visualization:** Matplotlib, Grad-CAM
- **Environment:** Google Colab (T4 GPU)

## Methodology
1. **Data Acquisition:**
   - Tabular data from King County housing dataset.
   - Satellite images fetched programmatically using **ESRI World Imagery** via Lat/Lon coordinates.
2. **Preprocessing:**
   - Tabular: Standard scaling (Z-score normalization).
   - Images: Resized to 224x224, normalized to ImageNet standards.
3. **Model Architecture:**
   - **Visual Branch:** ResNet18 (pretrained) extracts a 512-dim feature vector.
   - **Tabular Branch:** 3-layer MLP extracts a 128-dim feature vector.
   - **Fusion:** Vectors are concatenated -> Final Regression Head -> Price Prediction.

## Results
- **Model R² Score:** 0.846 (84.6%)
- **Validation RMSE:** 136,123
- **Performance:** The fusion model successfully integrates visual context, maintaining high accuracy even when introduced to noisy unstructured image data.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt` (or use the provided script).
3. Run the training pipeline:
   ```bash
   python train.py
