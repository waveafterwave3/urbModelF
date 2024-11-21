import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
from sklearn.preprocessing import LabelEncoder
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import spacy
import pickle

MODEL_NAME = "yagababa/UrbModel"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

with open('./label_classes.pkl', 'rb') as f:
    label_encoder = LabelEncoder()
    label_encoder.classes_ = pickle.load(f)

model_path = "street_modelF"
nlp = spacy.load(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = np.argmax(logits.cpu().numpy(), axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]
    return predicted_label

def detect_address(text):
    doc = nlp(text)
    address_entities = [ent.text for ent in doc.ents]
    return address_entities if address_entities else None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("k1k12k3.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    
    address = detect_address(user_message)
    
    category = classify_text(user_message)
    
    if address:
        response = f"Адрес: {', '.join(address)}\nКатегория: {category}"
    else:
        response = f"Адрес не найден\nКатегория: {category}"
    
    await update.message.reply_text(response)

def main():
    app = ApplicationBuilder().token("7694723703:AAHnO_0xjTCY0k-5eBqgfwf4qGosJ123ngs").build()

    app.add_handler(CommandHandler("start", start))
    
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling()

if __name__ == '__main__':
    main()
