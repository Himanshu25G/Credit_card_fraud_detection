# 💳 Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using the popular [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud). Handles extreme class imbalance with **SMOTE** and compares **Logistic Regression** vs **Random Forest**, deployed as an interactive **Streamlit** web app.

## 🚀 Overview

Out of 284,807 transactions in the dataset, only 492 are fraudulent (~0.17%) — a highly imbalanced classification problem. This project:

- Explores and preprocesses the transaction data
- Balances the training data using **SMOTE (Synthetic Minority Oversampling Technique)**
- Trains and compares two models: Logistic Regression and Random Forest
- Evaluates using confusion matrix, classification report, F1-score, and ROC-AUC
- Visualizes feature importance to understand what drives fraud predictions
- Ships a Streamlit app for real-time transaction prediction

## 🧠 How It Works

1. **Data Loading** – Loads `creditcard.csv` (284,807 transactions, 31 columns: `Time`, `V1`–`V28`, `Amount`, `Class`)
2. **Preprocessing** – Drops the `Time` column, scales `Amount` using `StandardScaler`
3. **Train/Test Split** – 80/20 stratified split to preserve class ratio
4. **Class Imbalance Handling** – Applies SMOTE on the training set to oversample the minority (fraud) class
5. **Model Training** – Trains Logistic Regression and Random Forest classifiers
6. **Evaluation** – Compares both models on confusion matrix, precision/recall/F1, and ROC-AUC
7. **Feature Importance** – Visualizes the top 10 most important features from the Random Forest model
8. **Deployment** – Random Forest model saved with `joblib` and served through a Streamlit app (`fraud.py`)

## 📊 Model Performance

| Metric | Logistic Regression | Random Forest |
|--------|---------------------|----------------|
| Precision (Fraud) | 0.0563 | 0.8710 |
| Recall (Fraud) | 0.9184 | 0.8265 |
| F1-score (Fraud) | 0.1061 | 0.8482 |
| ROC-AUC | 0.9700 | 0.9737 |
| Accuracy | 97.3% | 99.95% |

**Random Forest** was selected as the final model — it gives a far better precision/recall balance on the minority (fraud) class, which matters far more than raw accuracy on such an imbalanced dataset.

## 🗂️ Project Structure

```
Credit_card_fraud_detection/
├── creditcardFraudDetect.ipynb     # EDA, preprocessing, training, evaluation
├── creditcard_fraud_rf_model.pkl    # Trained Random Forest model
├── fraud.py                          # Streamlit app for live predictions
└── LICENSE                            # Apache 2.0 License
```

> **Note:** `creditcard.csv` is not included in this repo due to its large file size. Download it from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the project root before running the notebook.

## ⚙️ Tech Stack

- **Python**
- **pandas**, **numpy** – data handling
- **matplotlib**, **seaborn** – visualization
- **scikit-learn** – preprocessing, modeling, evaluation
- **imbalanced-learn (SMOTE)** – class imbalance handling
- **joblib** – model persistence
- **Streamlit** – web app for real-time predictions

## 🛠️ Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/Himanshu25G/Credit_card_fraud_detection.git
   cd Credit_card_fraud_detection
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib streamlit
   ```

3. **Add the dataset**
   Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the project root.

4. **Run the notebook**
   Open `creditcardFraudDetect.ipynb` to walk through EDA, preprocessing, training, and evaluation.

5. **Run the Streamlit app**
   ```bash
   streamlit run fraud.py
   ```
   Enter transaction feature values (`V1`–`V28` and `Amount`) to get a real-time fraud/genuine prediction.

## 🔮 Future Improvements

- Try gradient boosting models (XGBoost, LightGBM) for potentially better precision-recall trade-off
- Add SHAP-based explainability for individual predictions
- Deploy the Streamlit app to the cloud (e.g. Streamlit Community Cloud)
- Add threshold tuning to optimize the precision-recall trade-off for real-world use

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

## 👤 Author

**Himanshu Ranjan**
[GitHub](https://github.com/Himanshu25G) · [LinkedIn](https://linkedin.com/in/himanshu-ranjan-25g/)
