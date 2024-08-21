import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Simulated dataset
data = pd.DataFrame({
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'attendance': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    'previous_grades': [60, 65, 70, 75, 80, 85, 90, 95, 100, 105],
    'final_grade': [65, 70, 75, 80, 85, 90, 95, 100, 105, 110]
})

# Features and target
X = data[['study_hours', 'attendance', 'previous_grades']]
y = data['final_grade']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R-squared: {r2_score(y_test, y_pred)}")

# Display coefficients
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Display predictions
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
}).reset_index(drop=True)
print(results)
