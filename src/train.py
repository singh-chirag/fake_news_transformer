# # src/train.py

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from tqdm import tqdm

from src import config
from src.dataset import FakeNewsDataset
from src.model import FakeNewsClassifier


def main():
    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Load data
    df = pd.read_csv(config.DATA_PATH)

    texts = df[config.TEXT_COLUMN].astype(str).tolist()
    labels = df[config.LABEL_COLUMN].astype(int).tolist()



    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=config.RANDOM_STATE,
        stratify=labels
    )

    # Dataset & DataLoader
    train_dataset = FakeNewsDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    # Model
    model = FakeNewsClassifier()
    model.to(device)

    # Optimizer & Loss
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(config.EPOCHS):
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in loop:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "models/bert_fake_news.pt")
    print("Model saved to models/bert_fake_news.pt")


if __name__ == "__main__":
    main()

import os
os.makedirs("models", exist_ok=True)

torch.save(model.state_dict(), "models/bert_fake_news.pt")
print("Model saved to models/bert_fake_news.pt")
