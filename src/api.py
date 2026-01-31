# src/api.py
from src.logger import logger
import uuid

from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import FakeNewsPredictor

app = FastAPI(
    title="Fake News Detection API",
    version="1.0"
)

predictor = FakeNewsPredictor()


class NewsRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    note: str

@app.post("/predict", response_model=PredictionResponse)
def predict_news(req: NewsRequest):
    request_id = str(uuid.uuid4())

    if len(req.text.strip()) == 0:
        logger.warning(f"{request_id} | empty_input")
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    result = predictor.predict(req.text)

    logger.info(
        f"{request_id} | label={result['label']} "
        f"| confidence={result['confidence']} "
        f"| text_len={len(req.text.split())}"
    )

    return result


