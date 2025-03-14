import joblib
import pandas as pd

print("predict.py script is running...")

# Load the trained model, scaler, and column names
print("Loading model and scaler...")
model = joblib.load("random_forest_best_model.pkl")
scaler = joblib.load("scaler.pkl")
columns_used = joblib.load("columns_used.pkl")  # Load the correct feature names

print("Model and Scaler loaded successfully!")

# Create a test sample with the same structure as training data
new_data_dict = {
    'Age': 35,
    'DailyRate': 500,
    'DistanceFromHome': 10,
    'Education': 3,
    'EmployeeCount': 1,
    'EmployeeNumber': 101,
    'EnvironmentSatisfaction': 4,
    'Gender': 1,
    'HourlyRate': 60,
    'JobInvolvement': 3,
    'JobSatisfaction': 4,
    'MonthlyIncome': 5000,
    'MonthlyRate': 15000,
    'NumCompaniesWorked': 2,
    'OverTime': 1,
    'PercentSalaryHike': 12,
    'PerformanceRating': 3,
    'RelationshipSatisfaction': 3,
    'StandardHours': 80,
    'StockOptionLevel': 1,
    'TotalWorkingYears': 10,
    'TrainingTimesLastYear': 3,
    'WorkLifeBalance': 2,
    'YearsAtCompany': 5,
    'YearsInCurrentRole': 3,
    'YearsSinceLastPromotion': 2,
    'YearsWithCurrManager': 3,
    'BusinessTravel_Travel_Frequently': 0,
    'BusinessTravel_Travel_Rarely': 1,
    'Department_Research & Development': 1,
    'Department_Sales': 0,
    'EducationField_Life Sciences': 1,
    'EducationField_Marketing': 0,
    'EducationField_Medical': 0,
    'EducationField_Other': 0,
    'EducationField_Technical Degree': 0,
    'JobRole_Human Resources': 0,
    'JobRole_Laboratory Technician': 0,
    'JobRole_Manager': 0,
    'JobRole_Manufacturing Director': 0,
    'JobRole_Research Director': 0,
    'JobRole_Research Scientist': 0,
    'JobRole_Sales Executive': 1,
    'JobRole_Sales Representative': 0,
    'MaritalStatus_Married': 1,
    'MaritalStatus_Single': 0
}

# Convert to DataFrame
new_data = pd.DataFrame([new_data_dict])

# Ensure the columns match the training data
new_data = new_data.reindex(columns=columns_used, fill_value=0)
# Print feature names before scaling
print(f"Number of features in new_data: {new_data.shape[1]}")
print(f"Feature names in new_data: {list(new_data.columns)}")

# Print expected feature names from columns_used.pkl
print(f"Number of expected features: {len(columns_used)}")
print(f"Expected feature names: {list(columns_used)}")

# Check for missing and extra features
missing_features = set(columns_used) - set(new_data.columns)
extra_features = set(new_data.columns) - set(columns_used)

print(f"Missing features: {missing_features}")
print(f"Extra features: {extra_features}")

# Scale the data
new_data_scaled = scaler.transform(new_data)

# Make prediction
prediction = model.predict(new_data_scaled)

# Print the result
print("Predicted Attrition:", "Yes" if prediction[0] == 1 else "No")
# Print feature names before scaling
print(f"Number of features in new_data: {new_data.shape[1]}")
print(f"Feature names in new_data: {list(new_data.columns)}")

