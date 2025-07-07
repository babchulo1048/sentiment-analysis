# Install required libraries
!pip install transformers torch scikit-learn pandas spacy
!python -m spacy download en_core_web_sm

# Import libraries
import pandas as pd
import spacy
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# ✅ Step 1: Create Sample Dataset
data = {
    'text': [
        "I love the new Tesla Model Y! It's fantastic.",
        "Worst customer service ever. Tesla disappointed me.",
        "The autopilot feature is pretty cool!",
        "I'm so tired of Tesla's battery issues.",
        "Not sure about the design of the new model.",
        "Absolutely amazing experience driving Tesla!",
        "Terrible update, now my car keeps crashing.",
        "The Tesla showroom was impressive.",
        "Their support team helped me a lot, appreciated.",
        "Waste of money. I'm done with Tesla."
    ],
    'label': ['positive', 'negative', 'positive', 'negative', 'neutral',
              'positive', 'negative', 'positive', 'positive', 'negative']
}

df = pd.DataFrame(data)

# ✅ Step 2: Preprocess Text (Clean & Lemmatize)
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def preprocess(text):
    text = clean_text(text).lower()
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

df['cleaned_text'] = df['text'].apply(preprocess)

# ✅ Step 3: Encode Text Using BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

encoded_inputs = tokenizer(
    df['cleaned_text'].tolist(),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors='pt'
)

# ✅ Step 4: Convert Sentiment Labels into Numbers
le = LabelEncoder()
labels = le.fit_transform(df['label'])

# ✅ Step 5: Prepare Data for Training
input_ids = encoded_inputs['input_ids']
attention_mask = encoded_inputs['attention_mask']
labels = torch.tensor(labels)

# Split data into train and test sets with stratification
train_inputs, test_inputs, train_masks, test_masks, train_labels, test_labels = train_test_split(
    input_ids, attention_mask, labels, test_size=0.3, random_state=42, stratify=labels
)

# Create DataLoader for training
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# ✅ Step 6: Load BERT Model and Train
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
model.train()
for epoch in range(3):
    for batch in train_loader:
        b_input_ids, b_attention_mask, b_labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} completed")

# ✅ Step 7: Evaluate Model
model.eval()
test_dataset = TensorDataset(test_inputs, test_masks, test_labels)
test_loader = DataLoader(test_dataset, batch_size=2)

predictions = []
true_labels = []
with torch.no_grad():
    for batch in test_loader:
        b_input_ids, b_attention_mask, b_labels = [b.to(device) for b in batch]
        outputs = model(b_input_ids, attention_mask=b_attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(b_labels.cpu().numpy())

# Fix for classification report
unique_labels = np.unique(true_labels)
unique_label_names = [le.classes_[i] for i in unique_labels]

accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(true_labels, predictions, labels=unique_labels, target_names=unique_label_names))

# ✅ Step 8: Save Model and Tokenizer
model.save_pretrained("sentiment_model")
tokenizer.save_pretrained("sentiment_model")
le_classes = le.classes_.tolist()
pd.Series(le_classes).to_json("label_encoder.json")

# ✅ Step 9: Function to Predict Sentiment for New Input
def predict_sentiment(text):
    model.eval()
    cleaned_text = preprocess(text)
    encoded = tokenizer(cleaned_text, truncation=True, padding=True, max_length=128, return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        pred = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]
    return le.inverse_transform([pred])[0]
