from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_id = "ayushk1233/self-healing-sst2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

# Add a manual label mapping (SST-2 → 0 = negative, 1 = positive)
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
model.config.id2label = id2label
model.config.label2id = label2id

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        confidence, predicted_class = torch.max(probs, dim=1)

    label = model.config.id2label[predicted_class.item()]
    return label, confidence.item()

# Test examples
for text in [
    "The movie was painfully slow and boring.",
    "The movie was fantastic.",
    "it was an average movie"
]:
    label, conf = predict(text)
    print(f"Text: {text}")
    print(f"Predicted label: {label} | Confidence: {conf:.4f}")
    print()
