import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# --- Config ---
MODEL_NAME = "bert-base-cased"
final_model_path = "Checkpoints/Bert/fake_news_model.pth"

# --- Redefine the model class ---
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
        x = self.dropout(self.fc3(x))
        return x

# --- Rebuild & load trained weights ---
inference_model = DiseaseMythBuster(MODEL_NAME)
inference_model.load_state_dict(torch.load(final_model_path, map_location="cpu"))
inference_model.eval()

print("Model reloaded and ready for inference ✅")

import torch
from transformers import BertTokenizer
import torch.nn.functional as F

# --- Reload tokenizer ---
MODEL_NAME = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# --- Define prediction function ---
def predict(text, model, tokenizer, max_len=100, device="cpu"):
    model.eval()
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
        outputs = model(input_ids, attention_mask)
        probs = torch.sigmoid(outputs).cpu().numpy()[0][0]
        label = int(probs >= 0.5)
        return label, float(probs)

sample_text = "people tripping covidbs19its kind like false prophet epidemics not epidemic impeccable timing political stuff like epidemics time frame elections stfu not sneeze n cough people"
label, confidence = predict(sample_text, inference_model, tokenizer)
print("Prediction:", "Fake" if label == 0 else "Real", "| Confidence:", confidence)
