# src/evaluate.py

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src import config
from src.dataset import FakeNewsDataset
from src.model import FakeNewsClassifier


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load data
    df = pd.read_csv(config.DATA_PATH)

    texts = df[config.TEXT_COLUMN].astype(str).tolist()
    labels = df[config.LABEL_COLUMN].astype(int).tolist()

    # Same split as training
    _, X_val, _, y_val = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=config.RANDOM_STATE,
        stratify=labels
    )

    val_dataset = FakeNewsDataset(X_val, y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    # Load model
    checkpoint = torch.load("models/checkpoint.pt", map_location=device)

    model = FakeNewsClassifier().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()


    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
