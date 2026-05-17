import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

data = pd.read_csv("heart.csv")

# convert text columns to numbers
#data = pd.get_dummies(data)
data["Sex"] = data["Sex"].map({"M":1, "F":0})
data["ChestPainType"] = data["ChestPainType"].map({"TA":0,"ATA":1,"NAP":2,"ASY":3})
data["RestingECG"] = data["RestingECG"].map({"Normal":0,"ST":1,"LVH":2})
data["ExerciseAngina"] = data["ExerciseAngina"].map({"N":0,"Y":1})
data["ST_Slope"] = data["ST_Slope"].map({"Up":0,"Flat":1,"Down":2})

X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pickle.dump(model, open("heart_model.pkl", "wb"))

print("Heart model trained successfully")