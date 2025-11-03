# Fake Job Posting Detector with Personalized Risk Assessment and Explainability

## Project Overview

This project develops a robust system for detecting fraudulent job postings, powered by a fine-tuned DistilBERT model. Beyond basic classification, it introduces a novel **personalized risk assessment** mechanism that adapts to individual user profiles (e.g., 'fresher', 'experienced') and provides **explainable AI (XAI)** insights using SHAP, making the system transparent and user-centric.

The primary goal is to protect job seekers from sophisticated online scams by providing accurate, personalized, and interpretable risk scores for job advertisements.

## Key Features

* **High-Accuracy Fake Job Classification:** Utilizes a fine-tuned DistilBERT transformer model for state-of-the-art text classification, achieving excellent performance even on highly imbalanced datasets.
* **Personalized Risk Assessment (Stage 2):** Dynamically adjusts the job's risk score based on the user's career stage (e.g., 'fresher', 'mid-career', 'experienced') and the job's target experience level. This ensures that the system's warnings are tailored to the individual's vulnerability and context.
* **Explainable AI (XAI) with SHAP:** Provides word-level explanations for the model's predictions, showing exactly which parts of the job posting contribute to its "fake" or "real" classification. This fosters user trust and helps in understanding the underlying reasons for a risk assessment.
* **Comprehensive Evaluation:** Thoroughly evaluated on an unseen test set using metrics crucial for imbalanced datasets (Precision, Recall, F1-score for the minority class).
* **Production-Ready Architecture:** Designed with a clear separation of concerns, suitable for integration into web applications or job board platforms.

## System Architecture

The system operates in a two-stage pipeline:

1.  **Stage 1: Base Risk Model (DistilBERT Classifier):** Analyzes the raw job posting text to produce an initial `base_risk_probability` of the job being fraudulent.
2.  **Stage 2: Personalization Engine (Rule-Based Logic):** Takes the `base_risk_probability`, the job's extracted experience level, and the user's profile to compute a `personalized_risk_score`.
3.  **Explainability Layer (SHAP):** Generates explanations for the personalized risk score by explaining the output of the full pipeline.


## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Digaa2710/Sentinel-XAI.git](https://github.com/Digaa2710/Sentinel-XAI.git)
    cd Sentinel-XAI
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Ensure `requirements.txt` contains `tensorflow`, `transformers`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `shap`, `ipykernel` for notebooks)

4.  **Download the Dataset:**
    * Download `fake_job_postings.csv` from Kaggle: [https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
    * Place it in the `data/` directory.

5.  **Run Notebooks (Sequential Execution Recommended):**
    * Start Jupyter Lab or Jupyter Notebook:
        ```bash
        jupyter lab
        ```
    * Execute the notebooks in the `notebooks/` directory sequentially (from `1_data_preprocessing.ipynb` to `6_full_system_demo.ipynb`) to replicate the data processing, model training, evaluation, personalization, and explainability setup.

## Model Evaluation Results

The fine-tuned DistilBERT model was rigorously evaluated on an unseen test set (20% of the dataset, stratified split) with the following key results:

* **Overall Test Accuracy:** 0.9885 (98.85%)
* **Fake Job (Class 1) - Precision:** 0.93
* **Fake Job (1) - Recall:** 0.82
* **Fake Job (1) - F1-score:** 0.87
* **Confusion Matrix:**
    * True Negatives (Real correctly predicted as Real): 3393
    * False Positives (Real incorrectly predicted as Fake): 10
    * **False Negatives (Fake incorrectly predicted as Real): 31**
    * True Positives (Fake correctly predicted as Fake): 142

These scores demonstrate the model's high capability to detect fake job postings, achieving strong recall for the minority class while maintaining a low rate of false alarms.

## Personalization & Explainability in Action

The system provides not just a classification but a tailored experience:

1.  **Base Risk Prediction:** A job posting might receive a `base_risk_probability` of **8.20%** from DistilBERT.
2.  **Personalization:** If a `fresher` user views this job, and the job itself is also identified as `fresher`-level, the personalization engine could apply a higher adjustment factor (e.g., `1.8`), leading to a `personalized_risk_score` of around **14.76%** (8.20% * 1.8 = 14.76%).
3.  **Personalized SHAP Explanation:** SHAP would then explain *why* this 14.76% personalized risk was assigned, highlighting words in the job posting that contributed to this specific (adjusted) risk perception for the fresher user.

## Future Enhancements

* **Dynamic Personalization Weights:** Implement a small, lightweight learning model for the personalization engine (e.g., a shallow MLP) instead of fixed rules, allowing weights to adapt over time based on feedback.
* **More Sophisticated Feature Extraction:** Enhance job experience level extraction using more advanced NLP techniques or external metadata.


### Results 
##  SHAP Results

### 🔹 Result 1
![Result 1](SHAP%201.jpg)

### 🔹 Result 2 (SHAP)
![Result 2](SHAP%202.jpg)



