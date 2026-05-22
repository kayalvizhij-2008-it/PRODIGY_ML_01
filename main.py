import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv(r"C:\Users\kayal\Downloads\house-prices-advanced-regression-techniques\train.csv")

# Select features
X = data[["GrLivArea", "BedroomAbvGr", "FullBath"]]

# Target variable
y = data["SalePrice"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Check accuracy
score = r2_score(y_test, y_pred)

print("Model Accuracy:", score)

# Predict custom house price
sample_house = pd.DataFrame(
    [[2000, 3, 2]],
    columns=["GrLivArea", "BedroomAbvGr", "FullBath"]
)

predicted_price = model.predict(sample_house)

print("Predicted House Price:", predicted_price[0])