from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Load dataset
df = pd.read_csv("data/tune.csv")
input_texts = df['formal'].tolist()  # Replace with your column name
target_texts = df['informal'].tolist()  # Replace with your column name

# Dataset Class
class CustomDataset(Dataset):
    def __init__(self, tokenizer, inputs, targets, max_len=512):
        self.tokenizer = tokenizer
        self.inputs = inputs
        self.targets = targets
        self.max_len = max_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]

        input_encoding = tokenizer(
            input_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt"
        )
        target_encoding = tokenizer(
            target_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt"
        )

        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

# DataLoader
dataset = CustomDataset(tokenizer, input_texts, target_texts)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(3):  # number of epochs
    progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}")
    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix({'loss': loss.item()})
# Save the fine-tuned model
model_save_path = 'results/tune_model.pt'
torch.save(model.state_dict(), model_save_path)

# Function to generate a prediction
def generate_prediction(input_text, model, tokenizer, device):
    model.eval()
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(inputs, max_length=40, num_beams=5, early_stopping=True)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

device = 'cpu'
model_save_path = 'results/tune_model.pt'
# Test the model
model.load_state_dict(torch.load(model_save_path))
model.to(device)

test_sentence = "A man walks by a silver vehicle."
prediction = generate_prediction(test_sentence, model, tokenizer, device)
print("Predicted Translation:", prediction)