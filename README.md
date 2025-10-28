# Explainable & Personalized Fake Job Detection System

This project builds a machine learning system to identify fraudulent job postings. It goes beyond simple classification by providing **explainable insights** into *why* a job is flagged and **personalizes alerts** based on a job seeker's profile.

## 🎯 The Problem

With the rise of online job portals, fake job postings have become a major threat, often leading to data theft or financial loss.

Traditional detection systems only provide a binary "real" or "fake" label. This lack of transparency (the "black box" problem) reduces user trust and doesn't help job seekers learn to spot risky patterns.

Furthermore, these systems are "one-size-fits-all" and fail to consider a user's unique characteristics, such as their career stage or domain interest, which can make them more vulnerable to specific types of scams.

## ✨ Key Features

This system is built on three core pillars:

1.  **🕵️‍♂️ Accurate Detection:** A `RandomForestClassifier` trained on a mix of text and metadata features to accurately predict the probability of a job being fraudulent.
2.  **💡 Explainable AI (XAI):** Uses the **SHAP** (SHapley Additive exPlanations) library to explain *exactly which features* (e.g., "missing company profile," "vague requirements") contributed to a job's risk score.
3.  **👤 Personalized Alerts:** A rule-based engine that combines the model's prediction and the SHAP explanation to generate a custom, easy-to-understand warning tailored to user personas (e.g., "New Graduate," "Career Switcher").

## 🛠️ Technology Stack

* **Python 3**
* **Pandas:** For data loading and manipulation.
* **NLTK:** For natural language processing and text cleaning.
* **Scikit-learn:** For the ML pipeline (`TfidfVectorizer`, `RandomForestClassifier`, `ColumnTransformer`).
* **Imbalanced-learn:** For `SMOTE` to handle the highly imbalanced dataset.
* **SHAP:** For model explainability.
* **Jupyter / Kaggle Notebooks:** For development and analysis.

---

## 📈 Project Workflow

### 1. Data Preprocessing & Feature Engineering

* **Loaded Data:** The dataset contained ~18,000 job postings with 18 features, including a `fraudulent` target flag.
* **Text Cleaning:** All relevant text columns (`title`, `description`, `requirements`, etc.) were combined into a single `text_combined` feature. This text was then cleaned by lowercasing, removing stopwords, and stripping punctuation.
* **Feature Engineering:** New, high-signal features were created from the metadata:
    * `is_company_profile_missing`
    * `is_requirements_missing`
    * `is_benefits_missing`
    * `is_salary_range_missing`

### 2. Model Training

* **Mixed-Data Pipeline:** A `ColumnTransformer` was used to apply different preprocessing steps to different columns simultaneously:
    * **Text:** `TfidfVectorizer`
    * **Categorical:** `OneHotEncoder`
    * **Binary Flags:** `passthrough`
* **Handling Imbalance:** The dataset was highly imbalanced (~5% fake jobs). The `imblearn.Pipeline` was used to integrate **SMOTE** (Synthetic Minority Over-sampling TEchnique), which only oversamples the training data to prevent data leakage.
* **Model:** A `RandomForestClassifier` was chosen for its high performance and its compatibility with tree-based SHAP explainers.

### 3. Model Explainability (The "Why")

After training, a `shap.TreeExplainer` was used to analyze the model's decisions on a 1,000-job sample from the test set. This revealed the "global" logic of the model.

### 4. Personalization Engine (The "For Whom")

A function was built to:
1.  Take a job and a user persona (e.g., 'new_graduate').
2.  Get the model's risk score (e.g., 65%).
3.  Get the top 3 "red flags" from the SHAP explanation.
4.  Generate a custom alert that highlights the *specific risks* relevant to that user.

---

## 📊 Results & Demo

### Model Performance

The final model achieved excellent and, more importantly, *balanced* results on the unseen test data.

| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Real (0)** | 0.99 | 1.00 | 0.99 |
| **Fake (1)** | **0.97** | **0.71** | **0.82** |
| | | | |
| **Accuracy** | | | **0.99** |

The key metrics are for the "Fake" class:
* **Precision (0.97):** When the model flags a job as "Fake," it is correct 97% of the time. This builds user trust.
* **Recall (0.71):** The model successfully finds and catches 71% of all fake jobs.

### Global Explainability (The "Why")
**Key Insights from the Plot:**
* **Top Red Flags:** `required_experience_Missing` and `is_benefits_missing` were the strongest predictors of a fake job.
* **Keyword Signals:** Words like "services" and "work" (often found in vague "work-from-home" scams) pushed the risk score higher.
* **Trust Signals:** `employment_type_Full-time` was a strong signal for a *real* job, pushing the risk score lower.

### Demo: Personalized Alerts

Here are three examples of the final system in action:

#### 1. The Clearly FAKE Job

> **--- 🔔 Example Alert for a FAKE Job ---**
>
> **Job Title:** 'Home Based Data Entry/Data Typist'
>
> 🚨 **WARNING!** This job has a **100% risk score** of being FAKE.
>
> **Top Red Flags:**
> * Keyword: entry
> * employment_type: Part-time
> * required_experience is missing

#### 2. The Clearly REAL Job

> **--- ✅ Example Alert for a REAL Job ---**
>
> **Job Title:** 'Senior Java Developer'
>
> ✅ **Looks Safe.** This job has a very low risk score (0%).
>
> **Top Trust Signals:**
> * Keyword: java
> * employment_type: Full-time
> * Keyword: development

#### 3. The AMBIGUOUS Job

> **--- 🧐 Example Alert for an AMBIGUOUS Job ---**
>
> **Job Title:** 'KMC '
>
> ✅ **Looks Safe.** This job has a very low risk score (38%).
>
> **Top Trust Signals:**
> * program
> * services
> * work

*(Note: The 38% score is below the 50% threshold. The "trust signals" show that while "services" and "work" are *globally* red flags, for this *specific* job's context, the model found them to be legitimate signals, demonstrating a nuanced understanding.)*

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Digaa2710/Sentinel-XAI.git](https://github.com/Digaa2710/Sentinel-XAI.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas nltk scikit-learn imbalanced-learn shap
    ```
3.  **Place your data:** Add your `fake_job_postings.csv` (or `.xlsx`) file to the root directory.
4.  **Run the notebook:** Open and run the `.ipynb` notebook in Kaggle, Google Colab, or a local Jupyter instance.

The SHAP summary plot shows the top 20 features and their impact on the "fake" prediction.

**(Insert your SHAP summary plot screenshot here)**
