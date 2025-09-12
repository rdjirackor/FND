from flask import Flask, render_template, request
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import pandas as pd
import requests
import numpy as np
import pickle
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



# Paths
SVM_DIR = "Checkpoints/SVM"

MLP_PATH = "Checkpoints/MLP/Ensemble_MLP.pkl"

LOGREG_DIR = "Checkpoints/LogReg_Count"

RF_DIR = "Checkpoints/RF"

NB_DIR = "Checkpoints/NB"


logreg_model = joblib.load(os.path.join(LOGREG_DIR, "logreg_model.pkl"))
logreg_count_vectorizer = joblib.load(os.path.join(LOGREG_DIR, "count_vectorizer.pkl"))



mlp_model = joblib.load(MLP_PATH)


svm_tfidf = joblib.load(os.path.join(SVM_DIR, "svm_improved_tfidf.pkl"))
svm_count = joblib.load(os.path.join(SVM_DIR, "svm_improved_count.pkl"))
tfidf_vectorizer = joblib.load(os.path.join(SVM_DIR, "tfidf_improved_vectorizer.pkl"))
count_vectorizer = joblib.load(os.path.join(SVM_DIR, "count_improved_vectorizer.pkl"))


rf_model = joblib.load(os.path.join(RF_DIR, "rf_smote_tfidf.pkl"))







class WeightedNBDetector:
    def __init__(self, max_features=15000, ngram_range=(1,2)):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features, ngram_range=ngram_range)
        self.model = MultinomialNB()
        self.threshold = 0.5
        self.fitted = False

    def fit(self, texts, labels):
        raise NotImplementedError("Training is not supported inside Flask. Load pretrained model instead.")

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        probs = self.model.predict_proba(X)[:,1]
        return (probs >= self.threshold).astype(int)

    def predict_proba(self, texts):
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

nb_detector = joblib.load(os.path.join(NB_DIR, "weighted_nb_detector.pkl"))



# -------------------------
# 3. Prediction (BERT only for now)
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
        else:  # Count
            X = count_vectorizer.transform([text])
            score = svm_count.decision_function(X)[0]

        # Convert decision score → probability-like value
        prob = 1 / (1 + np.exp(-score))
        label = 1 if prob >= 0.5 else 0
        return label, float(prob)
    
    elif model_choice == "MLP":
        if vectorizer_choice.upper() == "TFIDF":
            X = tfidf_vectorizer.transform([text])
        else:  # Count
            X = count_vectorizer.transform([text])

        prob = mlp_model.predict_proba(X)[0][1]  # probability of "Real"
        label = 1 if prob >= 0.5 else 0
        return label, float(prob)

    elif model_choice == "LogReg":
    # Only Count Vectorizer was trained here
        X = logreg_count_vectorizer.transform([text])
        prob = logreg_model.predict_proba(X)[0][1]  # probability of "Real"
        label = 1 if prob >= 0.5 else 0
        return label, float(prob)
    
    elif model_choice == "NB":
    # Use the loaded WeightedNBDetector directly
        probs = nb_detector.predict_proba([text])[:, 1][0]  # probability of "Real"
        label = 1 if probs >= nb_detector.threshold else 0
        return label, float(probs)

    elif model_choice == "RF":
        prob = rf_model.predict_proba([text])[0][1]  # probability of "Real"
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
          "q": query, "num": num_results
           }
    try:
        response = requests.get(url, params=params)
        results = response.json()
        if "items" in results: 
            return [item["link"] for item in results["items"]] 
        else:
            # Print error info in console 
            print("Google API error:", results)
            return ["[No results or API misconfigured]"]
    except Exception as e:
        return [f"[Google API error: {e}]"]
    
   

# -------------------------
# 5. Feedback persistence
# -------------------------
def save_feedback(text, prediction, confidence, feedback):
    data = {
        "text": [text],
        "prediction": ["Real" if prediction == 1 else "Fake"],
        "confidence": [confidence],
        "user_feedback": [feedback]
    }
    df = pd.DataFrame(data)
    df.to_csv("feedback.csv", mode="a", header=False, index=False)

# -------------------------
# 6. Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    links = []
    if request.method == "POST":
        text = request.form["text_input"]

        # get values from the form
        model_choice = request.form.get("model", "BERT")  
        vectorizer_choice = request.form.get("vectorizer", "TFIDF")  

        # run prediction
        pred, confidence = predict_with_model(
            text,
            model_choice=model_choice,
            vectorizer_choice=vectorizer_choice
        )

        # package result
        result = {
            "prediction": "Real" if pred == 1 else "Fake",
            "confidence": round(confidence, 3),
            "text": text,
            "model": model_choice,
            "vectorizer": vectorizer_choice
        }

        # get google results
        links ="butterfly" #google_search(text, num_results=5)

    return render_template("index.html", result=result, links=links)

@app.route("/feedback", methods=["POST"])
def feedback():
    text = request.form["text"]
    prediction = request.form["prediction"]
    confidence = request.form["confidence"]
    feedback_value = request.form["feedback"]
    save_feedback(text, prediction, confidence, feedback_value)
    return "Thank you for your feedback!"

# -------------------------
# 7. Run
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)


