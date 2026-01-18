# Brain Tumor Diagnostic Classification

## Project Overview
This project implements a machine learning framework for the diagnostic classification of brain tumors. The system is designed to distinguish between **malignant** and **benign** tumors using physiological data features.

The core objective was to benchmark multiple supervised learning algorithms to identify the most reliable model for medical diagnostics, prioritizing **Recall (Sensitivity)** to minimize false negatives in a critical healthcare context.

## Key Results
| Model | Accuracy | Key Insight |
| :--- | :--- | :--- |
| **SVM (Scaled)** | **96.3%** | Achieved highest stability after feature standardization. |
| **Random Forest** | 94.1% | Strong baseline but slightly higher variance. |
| **KNN** | 89.5% | Sensitive to noise in high-dimensional feature space. |
| **Naive Bayes** | 85.2% | Underperformed due to feature independence assumptions. |

## Tech Stack
* **Language:** Python
* **Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Techniques:** Data Normalization (StandardScaler), K-Fold Cross-Validation, Confusion Matrix Analysis.

## Methodology
1.  **Data Preprocessing:**
    * Handled missing values and outliers.
    * Applied **Standard Scaler** to normalize feature distributions, which improved SVM performance from **61.9% to 96.5%** during training.
2.  **Model Selection:**
    * Benchmarked CART, Naive Bayes, KNN, and SVM.
    * Used **10-Fold Cross-Validation** to ensure model robustness and prevent overfitting.
3.  **Evaluation:**
    * Final model selected: **Support Vector Machine (SVM)**.
    * Test Set Performance: **96.3% Accuracy** (116 True Negatives, 65 True Positives).

## Usage
To run this project locally:
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Brain_Tumor_Classification.git](https://github.com/YOUR_USERNAME/Brain_Tumor_Classification.git)


2. Install dependencies:
   pip install pandas numpy scikit-learn matplotlib seaborn

3. Open the Jupyter Notebook:
   jupyter notebook "Brain Tumor Classification.ipynb"


4. Future Scope: 
    -> Integration of Deep Learning (CNNs) for MRI image-based classification.
    -> Deployment of the model via a Streamlit web interface for real-time predictions.   