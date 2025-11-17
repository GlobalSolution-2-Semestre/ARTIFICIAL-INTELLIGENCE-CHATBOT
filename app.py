# ======================================================
# MINDTRACK-IA - API Flask (Classificação e Regressão)
# ======================================================
# Pedro Henrique Luiz Alves Duarte RM 563405
# Guilherme Macedo Martins RM562396


from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model_burnout = pickle.load(open("modelo_burnout.pkl", "rb"))
model_prod = pickle.load(open("modelo_produtividade.pkl", "rb"))
scaler = pickle.load(open("scaler_prod.pkl", "rb"))

@app.route("/", methods=["GET"])
def home():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        fields = [
            "StressLevel", "SleepHours", "Workload",
            "ManagerSupport", "WorkLifeBalance", "PhysicalActivity"
        ]

        values = [float(request.form[f]) for f in fields]

        # Classificação
        pred_burnout = model_burnout.predict([values])[0]

        # Regressão
        X_scaled = scaler.transform([values])
        prod = float(model_prod.predict(X_scaled)[0])

        return render_template(
            "predict.html",
            burnout_risk=pred_burnout,
            productivity=round(prod, 2)
        )
    except Exception as e:
        return render_template("predict.html", error=str(e))

if __name__ == "__main__":
    app.run(port=8000, debug=True)
    
