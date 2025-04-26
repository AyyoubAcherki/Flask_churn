from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Charger le modèle
model = joblib.load(open("ch.joblib", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    CreditScore = int(data["CreditScore"])
    Geography = data["Geography"]
    Age = int(data["Age"])
    Tenure = int(data["Tenure"])
    Balance = float(data["Balance"])
    NumOfProducts = int(data["NumOfProducts"])
    HasCrCard = int(data["HasCrCard"])
    IsActiveMember = int(data["IsActiveMember"])
    EstimatedSalary = float(data["EstimatedSalary"])

    # One-hot Geography
    Geography_France = 1 if Geography == "France" else 0
    Geography_Germany = 1 if Geography == "Germany" else 0
    Geography_Spain = 1 if Geography == "Spain" else 0

    # One-hot Gender
    gender_value = data.get('Gender', None)
    if gender_value is None:
        return "Le champ 'Gender' est manquant", 400
    Gender_Female = 1 if gender_value == 'female' else 0
    Gender_Male = 1 if gender_value == 'male' else 0

    # Features vector
    features = np.array([[CreditScore, Age, Tenure, Balance, NumOfProducts,
                          HasCrCard, IsActiveMember, EstimatedSalary,
                          Geography_France, Geography_Germany, Geography_Spain,
                          Gender_Female, Gender_Male]])

    prediction = model.predict(features)[0]
    result = "Client à risque de départ" if prediction == 1 else "Client fidèle"

    return render_template('index.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

