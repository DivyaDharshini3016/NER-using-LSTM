# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
Build a Named Entity Recognition (NER) model that can automatically identify and classify entities like names of people, locations, organizations, and other important terms from text. The goal is to tag each word in a sentence with its corresponding entity label.
## DESIGN STEPS
### STEP 1
Import necessary libraries and set up the device (CPU or GPU).

### STEP 2
Load the NER dataset and fill missing values.

### STEP 3
Create word and tag dictionaries for encoding.

### STEP 4
Group words into sentences and encode them into numbers.

### STEP 5
Build a BiLSTM model for sequence tagging.

### STEP 6
Train the model using the training data.

### STEP 7
Evaluate the model performance on test data.


## PROGRAM
### Name: Divya Dharshini S
### Register Number: 212224240039
```python
class BiLSTMTagger(nn.Module):
    # Include your code here
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128):
        super(BiLSTMTagger, self).__init__()

        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=word2idx["ENDPAD"])
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, input_ids):
        embeds = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeds)
        logits = self.fc(lstm_out)
        return logits

model = BiLSTMTagger(len(word2idx), len(tag2idx)).to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=tag2idx["O"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids)
            loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids)
                loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print("-" * 30)

    return train_losses, val_losses
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

<img width="371" height="212" alt="image" src="https://github.com/user-attachments/assets/9a10f3ee-f6b0-4230-80bc-0fb1bae4c922" />
<img width="870" height="686" alt="image" src="https://github.com/user-attachments/assets/36a61bc9-b3c0-4ea1-a132-345f0deaca88" />

### Sample Text Prediction

<img width="1007" height="707" alt="image" src="https://github.com/user-attachments/assets/9cc563c6-fb28-4cea-93ea-2608e5a60ae8" />


## RESULT
The BiLSTM NER model achieved good accuracy in identifying entities like persons, locations, and organizations. It showed strong performance on frequent tags, with scope for improvement on rarer ones.
