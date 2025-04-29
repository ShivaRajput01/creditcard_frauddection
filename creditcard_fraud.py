# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, roc_curve

# 1. Load the data
df = pd.read_csv('creditcard.csv')

# Drop rows where 'Class' is NaN
df = df.dropna(subset=['Class'])

# Ensure 'Class' is integer
df['Class'] = df['Class'].astype(int)

print(f"Original dataset shape after dropping NaN in Class: {df.shape}")
print(df['Class'].value_counts(normalize=True))

# Scale 'Amount' and 'Time'
scaler = StandardScaler()
df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_Scaled'] = scaler.fit_transform(df[['Time']])

# Drop original 'Amount' and 'Time'
df.drop(['Amount', 'Time'], axis=1, inplace=True)

# Feature Engineering
df['High_Amount'] = df['Amount_Scaled'].apply(lambda x: 1 if x > 2 else 0)

# Prepare data
X = df.drop('Class', axis=1)
y = df['Class']


# Check for and handle NaN values in 'y' before train_test_split
print(f"Number of NaN values in 'y': {y.isnull().sum()}")  # Check for NaNs
y.dropna(inplace=True)  # Drop rows with NaN values in 'y'

# Reset index after dropping NaNs to avoid potential issues
X = X.reset_index(drop=True)

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Handle class imbalance using SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print(f"Resampled dataset shape: {X_train_res.shape}, {np.bincount(y_train_res)}")

# 7. Model Training: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_res, y_train_res)

# 8. Model Training: XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_res, y_train_res)

# 9. Evaluation Function
def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"{model_name} ROC-AUC Score: {roc_auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'{model_name} ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

# 10. Evaluate models
evaluate_model(rf_model, X_test, y_test, model_name="Random Forest")
evaluate_model(xgb_model, X_test, y_test, model_name="XGBoost")

# Optional: Save models
import joblib
joblib.dump(rf_model, 'rf_fraud_model.pkl')
joblib.dump(xgb_model, 'xgb_fraud_model.pkl')
