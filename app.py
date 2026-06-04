from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

data = pd.read_csv("diabetes_data.csv")

X = data[["Age", "BMI", "BloodPressure", "Glucose"]]
y = data["Risk"]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

@app.route("/")
def home():
    return "Diabetes Risk Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    user_data = request.get_json()

    age = user_data["Age"]
    bmi = user_data["BMI"]
    bp = user_data["BloodPressure"]
    glucose = user_data["Glucose"]

    new_data = pd.DataFrame(
        [[age, bmi, bp, glucose]],
        columns=["Age", "BMI", "BloodPressure", "Glucose"]
    )

    prediction = model.predict(new_data)[0]

    return jsonify({
        "Predicted_Diabetes_Risk": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)