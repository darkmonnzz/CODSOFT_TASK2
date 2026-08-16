# CodSoft Task 2: Credit Card Fraud Detection

## 📌 Overview
This project builds a machine learning model to detect fraudulent credit card transactions using a simulated transactions dataset. Given the highly imbalanced nature of fraud data (~0.58% fraud cases), the focus was on comparing multiple models and selecting the one with the best recall–precision balance for the minority (fraud) class.

## 📂 Dataset
Simulated credit card transaction dataset with separate train and test files.
- **Train set:** 1,296,675 rows
- **Test set:** 555,719 rows
- **Target column:** `is_fraud` (0 = legitimate, 1 = fraud)
- **Class distribution:** 99.42% legitimate, 0.58% fraud (highly imbalanced)

Dataset source: [Kaggle - Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

> Note: Due to GitHub's file size limits, the dataset CSV files are not included in this repo. Please download them from the Kaggle link above.

## 🛠️ Feature Engineering
- **Age** — derived from date of birth (`dob`) and transaction timestamp
- **Hour** and **Day of Week** — extracted from transaction timestamp
- **Distance** — Haversine distance between customer location and merchant location
- **Category** — one-hot encoded (14 transaction categories)
- **Gender** — binary encoded
- Dropped identifier/PII columns (`cc_num`, `first`, `last`, `street`, `trans_num`, etc.) with no predictive value

## 🤖 Models Trained
| Model | Precision (Fraud) | Recall (Fraud) | F1-score (Fraud) | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.02 | 0.74 | 0.04 | 0.908 |
| Decision Tree | 0.77 | 0.78 | 0.78 | 0.889 |
| **Random Forest** | **0.84** | **0.81** | **0.82** | **0.984** |

All models were trained with `class_weight='balanced'` to account for class imbalance. Features were scaled using `StandardScaler` for Logistic Regression (tree-based models don't require scaling).

## 🏆 Best Model: Random Forest
Random Forest achieved the best overall performance with the highest F1-score and ROC-AUC, making it the most reliable model for distinguishing fraudulent from legitimate transactions in this dataset.

## 📊 Evaluation Metric Choice
Accuracy is not a meaningful metric here due to extreme class imbalance — a model predicting "not fraud" for everything would still score ~99.4% accuracy. Instead, **precision, recall, F1-score, and ROC-AUC** were used to properly evaluate fraud-detection performance.

## 🧰 Tech Stack
- Python
- pandas, numpy
- scikit-learn (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier)

## 🚀 How to Run
1. Download the dataset from the Kaggle link above
2. Place `train.csv` and `test.csv` in the project folder
3. Run `fraud.ipynb`

## ✍️ Author
Purva Jain — [GitHub](https://github.com/darkmonnzz) | [LinkedIn](https://linkedin.com/in/purva-jain-83901b286)
