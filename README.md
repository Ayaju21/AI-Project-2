# 📰 News Headline Classification: Logistic Regression vs. Decision Tree

A Machine Learning project focused on Natural Language Processing (NLP) to classify news headlines into four categories: **World (0), Sports (1), Business (2), and Sci/Tech (3)**. 

## 🎯 Objective
The goal of this project is to build an end-to-end pipeline to preprocess textual data, extract features using TF-IDF, and compare the performance of **Logistic Regression** and **Decision Tree** models.

---

## 🛠️ Tech Stack & Tools
* **Language:** [e.g., Python]
* **Libraries:** * `Scikit-learn` (Modeling & Evaluation)
  * `Pandas` & `NumPy` (Data Manipulation)
  * `NLTK` or `SpaCy` (Text Preprocessing)
  * `Matplotlib` / `Seaborn` (Data Visualization)
---

## 🧬 Project Pipeline

### 1. Data Preprocessing
* **Text Cleaning:** Removing punctuation, special characters, and numbers.
* **Tokenization:** Breaking down headlines into individual words.
* **Stop-words Removal:** Eliminating common words (e.g., "the", "is") that don't add semantic value.
* **Feature Extraction:** Converting text into numerical vectors using **TF-IDF Vectorizer** (Term Frequency-Inverse Document Frequency).

### 2. Model Training
We implemented and trained two distinct algorithms:
* **Logistic Regression:** A linear model used for multi-class classification.
* **Decision Tree:** A non-linear model that splits data based on feature importance.

### 3. Evaluation Metrics
Models were compared using a full suite of classification metrics:
* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
