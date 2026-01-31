# src/config.py

MODEL_NAME = "distilbert-base-uncased"

# Data
DATA_PATH = "data/raw/fake_or_real_news.csv"
TEXT_COLUMN = "title"
LABEL_COLUMN = "real"


BATCH_SIZE = 16   # or 32 if RAM allows
MAX_LEN = 128     # titles do NOT need 512
EPOCHS = 1
LEARNING_RATE = 2e-5

RANDOM_STATE = 42
DEVICE = "cuda"
 