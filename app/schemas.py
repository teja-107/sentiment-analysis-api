from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, examples=["This movie was fantastic!"])


class PredictResponse(BaseModel):
    label: str
    confidence: float
