# ======================================================
# MINDTRACK-IA - API Flask (Classificação e Regressão)
# ======================================================
# Pedro Henrique Luiz Alves Duarte
# Guilherme Macedo Martins
# Descrição:
# API Flask que expõe dois modelos de IA:
#   - Classificação: Predição de risco de Burnout
#   - Regressão: Predição do nível de Produtividade
# ======================================================

import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# ======================================================
# Inicialização da aplicação Flask
# ======================================================
app = Flask(__name__)

# ======================================================
# Funções auxiliares
# ======================================================

def load_pickle(path):
    """Carrega um arquivo pickle de forma segura."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        print(f"⚠️ Arquivo não encontrado: {path}")
        return None

def response_ok(message, **extra):
    """Retorna uma resposta JSON de sucesso."""
    return jsonify({"status": "ok", "message": message, **extra}), 200

def response_error(message, **extra):
    """Retorna uma resposta JSON de erro."""
    return jsonify({"status": "error", "message": message, **extra}), 400

# ======================================================
# Carregamento dos modelos
# ======================================================

CLASSIFICATION_PATH = "modelo_classificacao_burnoutrisk.pkl"
REGRESSION_PATH = "modelo_regressao_xgboost.pkl"

classification_model = load_pickle(CLASSIFICATION_PATH)
regression_model = load_pickle(REGRESSION_PATH)

print("🧠 Status de carregamento dos modelos:")
print(f" - Classificação carregado: {classification_model is not None}")
print(f" - Regressão carregado: {regression_model is not None}")

# ======================================================
# Rotas básicas
# ======================================================

@app.route("/", methods=["GET"])
def home():
    """Rota inicial da API."""
    return response_ok("API Flask para Burnout (classificação) e Produtividade (regressão).")

@app.route("/health", methods=["GET"])
def health_check():
    """Verifica se a API e os modelos estão ativos."""
    return jsonify({
        "status": "ok",
        "message": "alive",
        "classification_loaded": classification_model is not None,
        "regression_loaded": regression_model is not None
    })

# ======================================================
# Rota de predição - Classificação (Burnout)
# ======================================================

@app.route("/predict/classification", methods=["POST"])
def predict_classification():
    """Predição de risco de Burnout."""
    if classification_model is None:
        return response_error("Modelo de classificação não carregado.")

    data = request.get_json()
    if not data:
        return response_error("Nenhum dado recebido.")

    try:
        features = np.array(data["features"]).reshape(1, -1)
        prediction = classification_model["model"].predict(features)[0]
        return response_ok("Predição realizada com sucesso.", prediction=int(prediction))
    except Exception as e:
        return response_error(f"Erro ao realizar predição: {e}")

# ======================================================
# Rota de predição - Regressão (Produtividade)
# ======================================================

@app.route("/predict/regression", methods=["POST"])
def predict_regression():
    """Predição de nível de produtividade."""
    if regression_model is None:
        return response_error("Modelo de regressão não carregado.")

    data = request.get_json()
    if not data:
        return response_error("Nenhum dado recebido.")

    try:
        features = np.array(data["features"]).reshape(1, -1)
        prediction = regression_model["model"].predict(features)[0]
        return response_ok("Predição realizada com sucesso.", prediction=float(prediction))
    except Exception as e:
        return response_error(f"Erro ao realizar predição: {e}")

# ======================================================
# Inicialização da API
# ======================================================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
