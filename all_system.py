import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset (you can replace this with your own dataset)
data = pd.read_csv('loan_dataset.csv')

# Data preprocessing
data.dropna(inplace=True)
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Encode categorical features (you may need more advanced encoding methods)
X = pd.get_dummies(X, drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a machine learning model (Random Forest Classifier in this case)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Example of making a prediction for a new applicant
new_applicant = pd.DataFrame({
    'Credit_Score': [750],
    'Income': [60000],
    'Loan_Amount': [200000],
    'Loan_Term_360': [1],
    'Married_Yes': [1],
    'Education_Graduate': [1]
})
new_applicant = pd.get_dummies(new_applicant, drop_first=True)
prediction = model.predict(new_applicant)

if prediction[0] == 'Y':
    print("Loan Approved")
else:
    print("Loan Denied")
