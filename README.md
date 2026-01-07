# Satellite Imagery-Based Property Valuation (SIBPV)

* **Author:** Dara Zenas Zephaniah
* **Enrollment Number:** 23321010

## Project Overview
This project implements a **Multimodal Regression Pipeline** that predicts property prices by fusing tabular real estate data with satellite imagery. By processing visual environmental context (greenery, road density) alongside traditional housing features, the model captures "curb appeal" value drivers that pure tabular models miss.

## Repository Structure 
* **`src/`**: Folder containing modular source code (`baseline.py`,`config.py`,`data_fetcher.py`,`datasets.py`,`gradcam.py`,`model.py`)
* **`23321010_final.csv`**: The final prediction file containing `id` and `predicted price`.
* **`23321010_report.pdf`**
* **`Full_Project_Implementation.ipynb`**
* **`baseline.py`**: A script reproducing the Tabular-Only XGBoost baseline results.
* **`data_fetcher.py`**: The script used to download images from the ESRI API.
* **`model_training.ipynb`**: The training loop for the multimodal (Fusion) model.
* **`preprocessing.ipynb`**: Data cleaning, feature engineering, and scaling logic.

## Setup Instructions 

### 1. Prerequisites
* **Python 3.10+**
* **Recommended:** GPU environment (Google Colab T4 or local CUDA).

### 2. Installation
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/ZenasZephaniah/SIBPV.git](https://github.com/ZenasZephaniah/SIBPV.git)
cd SIBPV
pip install torch torchvision pandas numpy opencv-python matplotlib scikit-learn tqdm xgboost openpyxl requests
```
### 3. How to Run Code
**Step 1: Download Satellite Images** Run the fetcher script to populate the `data/satellite` directory. It uses intelligent caching to avoid re-downloading existing images.
```
python data_fetcher.py
```
**Step 2: Preprocessing & Training** You can run the full pipeline using the main training script. The script handles data loading, preprocessing(scaling),model training, and generates the final `23321010_final.csv` in the `outputs/` folder (or root, depending on config)
```
python train.py
````
**Step 3: Verify Baseline Results** To reduce the comparison metrics cited in the report (Tabular-Only vs Multimodal), run the baseline script:
```
python baseline.py
```

