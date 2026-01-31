# src/inference.py

class FakeNewsPredictor:
    """
    Stable production-safe predictor.
    This is a STUB (no ML model loaded).
    """

    def __init__(self):
        pass

    def predict(self, text: str) -> dict:
        text = text.strip()

        if not text:
            return {
                "label": "UNKNOWN",
                "confidence": 0.0,
                "note": "empty_input"
            }

        # Simple heuristic (STUB LOGIC)
        word_count = len(text.split())

        if word_count < 6:
            label = "FAKE"
        else:
            label = "REAL"

        return {
            "label": label,
            "confidence": 0.55,
            "note": "stub_prediction_no_ml_model"
        }
