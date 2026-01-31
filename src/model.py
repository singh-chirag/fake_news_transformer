# src/model.py

import torch.nn as nn
from transformers import AutoModel
from src import config


class FakeNewsClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert = AutoModel.from_pretrained(config.MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            2
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(pooled_output)

        return self.classifier(x)
