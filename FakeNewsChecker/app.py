from flask import Flask, render_template, request
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import pandas as pd
import requests
import numpy as np
import joblib
import os


# -------------------------
# 1. Flask app
# -------------------------
app = Flask(__name__)

# -------------------------
# 2. Load models
# -------------------------
MODEL_NAME = "bert-base-cased"
MODEL_PATH = "Checkpoints/Bert/fake_news_model.pth"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

class DiseaseMythBuster(nn.Module):
    def __init__(self, model_name):
        super(DiseaseMythBuster, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(self.relu(self.fc1(pooled_output)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

bert_model = DiseaseMythBuster(MODEL_NAME)
bert_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
bert_model.eval()

SVM_DIR = "Checkpoints/SVM"
svm_tfidf = joblib.load(os.path.join(SVM_DIR, "svm_improved_tfidf.pkl"))
svm_count = joblib.load(os.path.join(SVM_DIR, "svm_improved_count.pkl"))
tfidf_vectorizer = joblib.load(os.path.join(SVM_DIR, "tfidf_improved_vectorizer.pkl"))
count_vectorizer = joblib.load(os.path.join(SVM_DIR, "count_improved_vectorizer.pkl"))


# -------------------------
# 3. Prediction
# -------------------------
def predict_with_model(text, model_choice="BERT", vectorizer_choice="TFIDF", max_len=100, device="cpu"):
    if model_choice == "BERT":
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        with torch.no_grad():
            outputs = bert_model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs).cpu().numpy()[0][0]
            label = 1 if probs >= 0.5 else 0
            return label, float(probs)

    elif model_choice == "SVM":
        if vectorizer_choice.upper() == "TFIDF":
            X = tfidf_vectorizer.transform([text])
            score = svm_tfidf.decision_function(X)[0]
        else:
            X = count_vectorizer.transform([text])
            score = svm_count.decision_function(X)[0]
        prob = 1 / (1 + np.exp(-score))
        label = 1 if prob >= 0.5 else 0
        return label, float(prob)

    return 0, 0.5


# -------------------------
# 4. Google Search
# -------------------------
GOOGLE_API_KEY = "AIzaSyA0pla-m_gxPFGnRgnmh8txxc4SqJEs8_Y"
SEARCH_ENGINE_ID = "b7183223587c14fe2"

def google_search(query, num_results=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": num_results
    }
    try:
        response = requests.get(url, params=params)
        results = response.json()
        if "items" in results:
            return [item["link"] for item in results["items"]]
        else:
            print("Google API error:", results)
            return ["[No results or API misconfigured]"]
    except Exception as e:
        return [f"[Google API error: {e}]"]


# -------------------------
# 5. Facticity Fact Checker Integration
# -------------------------
FACTICITY_API_KEY = "d9a0f52a-9f3d-472a-bb23-b85fabb0a3de"
FACTICITY_HEADERS = {
    "X-API-KEY": FACTICITY_API_KEY,
    "Content-Type": "application/json"
}

# -------------------------
# 9. Facticity API Integration
# -------------------------
FACTICITY_API_KEY = "d9a0f52a-9f3d-472a-bb23-b85fabb0a3de"
FACTICITY_URL = "https://api.facticity.ai/fact-check"

def get_fact_checker_results(text, version="v3", timeout=60, mode="sync"):
    headers = {
        "X-API-KEY": FACTICITY_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "query": text,
        "version": version,
        "timeout": timeout,
        "mode": mode
    }
    try:
        response = requests.post(FACTICITY_URL, json=payload, headers=headers, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            # Ensure sources are in proper format
            formatted_sources = []
            for s in data.get("sources", []):
                # Sometimes sources come as JSON strings
                if isinstance(s, str):
                    try:
                        s_dict = eval(s)  # convert string dict to actual dict safely
                        formatted_sources.append(s_dict)
                    except:
                        continue
                elif isinstance(s, dict):
                    formatted_sources.append(s)
            data["sources"] = formatted_sources
            return data
        else:
            print("Fact Checker API Error:", response.status_code, response.text)
    except Exception as e:
        print("Fact Checker Exception:", e)
    return None


# -------------------------
# 7. Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    links = []

    if request.method == "POST":
        text = request.form["text_input"]

        model_choice = request.form.get("model", "BERT")  
        vectorizer_choice = request.form.get("vectorizer", "TFIDF")  

        # -------------------------
        # 1. Model Prediction
        # -------------------------
        pred, confidence = predict_with_model(
            text,
            model_choice=model_choice,
            vectorizer_choice=vectorizer_choice
        )
        confidence_percent = round(confidence * 100, 1)
        prediction_text = "Real" if pred == 1 else "Fake"

        # -------------------------
        # 2. Facticity Fact-Checker
        # -------------------------
        fact_checker_results = get_fact_checker_results(text)

        # -------------------------
        # 3. Optional: Related Google Results
        # -------------------------
        if fact_checker_results and "sources" in fact_checker_results:
            for s in fact_checker_results["sources"]:
                try:
                    source_link = s.get("link") if isinstance(s, dict) else None
                    if source_link:
                        links.append(source_link)
                except Exception:
                    continue

        # -------------------------
        # 4. Aggregate results
        # -------------------------
        result = {
            "prediction": prediction_text,
            "confidence": confidence_percent,
            "text": text,
            "model": model_choice,
            "vectorizer": vectorizer_choice,
            "fact_checker": fact_checker_results  # Facticity API results
        }

    return render_template("index.html", result=result, links=links)




@app.route("/feedback", methods=["POST"])
def feedback():
    text = request.form["text"]
    prediction = request.form["prediction"]
    confidence = request.form["confidence"]
    feedback_value = request.form["feedback"]
    return "Thank you for your feedback!"


# -------------------------
# 8. Run
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
