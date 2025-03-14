#  Employee Attrition Prediction

##  Overview
This project predicts employee attrition using **machine learning** techniques.  
We trained multiple models and optimized a **Random Forest Classifier**, achieving an **F1-score of 0.9272**.

##  Dataset
- **Source**: [Kaggle HR Employee Attrition Dataset](https://www.kaggle.com/)  
- **Features**: Employee demographics, job satisfaction, salary details  
- **Target Variable**: `Attrition` (Yes/No)

##  Machine Learning Models Used
-  Logistic Regression  
-  Random Forest  (Best Model)  
-  XGBoost  

##  Model Optimization
- **Data Preprocessing**: Handled missing values, categorical encoding, feature scaling  
- **Feature Engineering**: Identified and removed multicollinear features  
- **SMOTE**: Balanced the dataset for better predictions  
- **Hyperparameter Tuning**: Used `GridSearchCV` to find the best parameters for Random Forest

## Final Results
| Model              | Accuracy | Precision | Recall | F1-score |
|-------------------|----------|-----------|--------|-----------|
| **Random Forest ** | **0.929** | **0.953** | **0.903** | **0.927** |
| XGBoost          | 0.927    | 0.957     | 0.895  | 0.925     |
| Logistic Reg.    | 0.881    | 0.902     | 0.854  | 0.877     |

## How to Use
1️⃣ **Install dependencies:**  
```bash
pip install -r requirements.txt
