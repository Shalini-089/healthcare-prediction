from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import pickle
import numpy as np
   

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///healthcare.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False 
db = SQLAlchemy(app)

#database table
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    disease = db.Column(db.String(50))
    result = db.Column(db.String(50))
    

# Load model & scaler
model = pickle.load(open("model/diabetes_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
heart_model = pickle.load(open("model/heart_model.pkl", "rb"))
parkinsons_model = pickle.load(open("model/parkinsons_model.pkl", "rb"))

#home route
@app.route("/")
def home():
    return render_template("index.html")

#diabetes page route
@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")

#diabetes prediction route
@app.route("/predict", methods=["POST"])
def predict():
    features = [
        float(request.form["Pregnancies"]),
        float(request.form["Glucose"]),
        float(request.form["BloodPressure"]),
        float(request.form["SkinThickness"]),
        float(request.form["Insulin"]),
        float(request.form["BMI"]),
        float(request.form["DiabetesPedigreeFunction"]),
        float(request.form["Age"])
    ]

    scaled = scaler.transform([features])
    prediction = model.predict(scaled)
    probability = model.predict_proba(scaled)[0][prediction[0]]

    confidence = round(probability * 100, 2)

    if prediction[0] == 1:
        result = "Diabetic"
        risk = "High Risk"
        color = "danger"
    else:
        result = "Not Diabetic"
        risk = "Low Risk"
        color = "success"

    #SAVE TO DATABASE
    new_data = Prediction(disease="Diabetes", result=result)
    db.session.add(new_data)
    db.session.commit()

    return render_template("diabetes.html",
                           prediction=result,
                           risk=risk,
                           confidence=confidence,
                           color=color)

#heart disease page route
@app.route("/heart")
def heart():
    return render_template("heart.html")

#heart disease prediction route
@app.route("/heart_predict", methods=["POST"])
def heart_predict():
    features = [
        float(request.form["age"]),
        float(request.form["sex"]),
        float(request.form["cp"]),
        float(request.form["trestbps"]),
        float(request.form["chol"]),
        float(request.form["fbs"]),
        float(request.form["restecg"]),
        float(request.form["thalach"]),
        float(request.form["exang"]),
        float(request.form["oldpeak"]),
        float(request.form["slope"])
    ]
    prediction = heart_model.predict([features])

    if prediction[0] == 1:
        result = "Heart Disease Detected"
        risk = "High Risk"
        color = "danger"
    else:
        result = "No Heart Disease"
        risk = "Low Risk"
        color = "success"

    # SAVE TO DATABASE
    new_data = Prediction(disease="Heart", result=result)
    db.session.add(new_data)
    db.session.commit()

    return render_template("heart.html",
                            prediction=result,
                            risk=risk,
                            color=color)    

#parkinsons page route
@app.route("/parkinsons")
def parkinsons():
    return render_template("parkinsons.html")

#parkinsons prediction route
@app.route("/parkinsons_predict", methods=["POST"])
def parkinsons_predict():
    features = [float(x) for x in request.form.values()]
    prediction = parkinsons_model.predict([features])

    if prediction[0] == 1:
        result = "Parkinson's Disease Detected"
        risk = "High Risk"
        color = "danger"
    else:
        result = "No Parkinson's Disease"
        risk = "Low Risk"
        color = "success"

    #SAVE TO DATABASE 
    new_data = Prediction(disease="Parkinson's", result=result)
    db.session.add(new_data)
    db.session.commit()

    return render_template("parkinsons.html",
                            prediction=result,
                            risk=risk,
                            color=color)    

#histrory page route
@app.route("/history")
def history():
    all_data = Prediction.query.all()
    return render_template("history.html", data=all_data)

if __name__ == "__main__":
    app.run(debug=True) 