from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    probability_spam: float | None = None
    probability_ham: float | None = None
