# creditcard_frauddection

This project detects fraudulent credit card transactions using Random Forest and XGBoost classifiers, with a focus on minimizing false positives while maintaining high recall.

## Dataset
- Credit card transactions dataset with anonymized features (V1–V28), Time, Amount, and Class labels.

## Steps
- Data Cleaning: Dropped NaN values.
- Preprocessing: Scaled Amount and Time, engineered High_Amount feature.
- Imbalanced Handling: Applied SMOTE oversampling.
- Modeling: Random Forest and XGBoost.
- Evaluation: Confusion Matrix, Classification Report, ROC-AUC.
- Saved final models for deployment.

## Result
- Achieved high ROC-AUC scores and minimized false positives significantly.


