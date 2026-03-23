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
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128):
        super(BiLSTMTagger, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size + 1,
            embedding_dim,
            padding_idx=word2idx["ENDPAD"]
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
```
```
model = BiLSTMTagger(len(word2idx), len(tag2idx)).to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=tag2idx["PAD"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```
```
# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, epochs=10):
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

            loss = loss_fn(
                outputs.view(-1, len(tag2idx)),
                labels.view(-1)
            )

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

                loss = loss_fn(
                    outputs.view(-1, len(tag2idx)),
                    labels.view(-1)
                )

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
  return train_losses, val_losses        
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

<img width="537" height="241" alt="image" src="https://github.com/user-attachments/assets/06e53990-6db5-4c48-bcda-b3b499f3c09a" />
<img width="542" height="622" alt="image" src="https://github.com/user-attachments/assets/efca42af-e835-43dd-b439-1109393719a8" />



<img width="513" height="612" alt="image" src="https://github.com/user-attachments/assets/ef918b86-10e1-4cea-be4e-1f5240939c5f" />


## RESULT
The BiLSTM NER model achieved good accuracy in identifying entities like persons, locations, and organizations. It showed strong performance on frequent tags, with scope for improvement on rarer ones.
