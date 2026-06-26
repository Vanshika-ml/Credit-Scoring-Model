import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split

from src.data_preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.train_model import train_model
from src.evaluate_model import evaluate

# Load dataset
df = pd.read_csv("data/credit_data.csv")

# Preprocessing
df = preprocess_data(df)

# Feature Engineering
df = create_features(df)

# Features and Target
X = df.drop("Credit_Score", axis=1)

y = df["Credit_Score"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Model
model = train_model(X_train, y_train)

#Save model
joblib.dump(model,"models/credit_model.pkl")
print("Model Saved Successfully!")

# Evaluate
evaluate(model, X_test, y_test)

importance = model.feature_importances_

plt.figure(figsize=(8,5))
plt.barh(X.columns,importance)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("Feature_importance.png")
plt.show()