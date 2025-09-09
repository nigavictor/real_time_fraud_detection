# Fraud Detection Model Evaluation Report

**Generated on:** 2025-09-09 15:17:42

**Number of models evaluated:** 4

## Model Performance Summary

| Model | ROC AUC | F1 Score | Precision | Recall | Fraud Detection Rate |
|-------|---------|----------|-----------|--------|---------------------|
| logistic_regression | 0.815 | 0.385 | 0.276 | 0.641 | 0.641 |
| random_forest | 0.854 | 0.502 | 0.522 | 0.484 | 0.484 |
| xgboost | 0.867 | 0.342 | 0.214 | 0.863 | 0.863 |
| ensemble | 0.872 | 0.492 | 0.422 | 0.589 | 0.589 |

## Detailed Model Results

### Logistic Regression

**Performance Metrics:**
- ROC AUC: 0.815
- Average Precision: 0.548
- F1 Score: 0.385
- Precision: 0.276
- Recall: 0.641
- Fraud Detection Rate: 0.641
- False Alarm Rate: 0.187

**Confusion Matrix:**
```
                Predicted
Actual    Legit  Fraud
Legit      7315   1685
Fraud       359    641
```

### Random Forest

**Performance Metrics:**
- ROC AUC: 0.854
- Average Precision: 0.590
- F1 Score: 0.502
- Precision: 0.522
- Recall: 0.484
- Fraud Detection Rate: 0.484
- False Alarm Rate: 0.049

**Confusion Matrix:**
```
                Predicted
Actual    Legit  Fraud
Legit      8556    444
Fraud       516    484
```

### Xgboost

**Performance Metrics:**
- ROC AUC: 0.867
- Average Precision: 0.616
- F1 Score: 0.342
- Precision: 0.214
- Recall: 0.863
- Fraud Detection Rate: 0.863
- False Alarm Rate: 0.353

**Confusion Matrix:**
```
                Predicted
Actual    Legit  Fraud
Legit      5821   3179
Fraud       137    863
```

### Ensemble

**Performance Metrics:**
- ROC AUC: 0.872
- Average Precision: 0.627
- F1 Score: 0.492
- Precision: 0.422
- Recall: 0.589
- Fraud Detection Rate: 0.589
- False Alarm Rate: 0.090

**Confusion Matrix:**
```
                Predicted
Actual    Legit  Fraud
Legit      8194    806
Fraud       411    589
```

## Recommendations

**Best Overall Model (ROC AUC):** ensemble (ROC AUC: 0.872)
**Best Balanced Model (F1):** random_forest (F1: 0.502)