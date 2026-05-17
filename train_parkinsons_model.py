import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

# Load dataset
data = pd.read_csv("parkinsons.csv")

# Remove name column if present
if 'name' in data.columns:
    data = data.drop('name', axis=1)

# Input and output
X = data.drop("status", axis=1)
y = data["status"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = SVC()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("parkinsons_model.pkl", "wb"))

print("Parkinson's model trained successfully")