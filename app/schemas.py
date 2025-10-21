from pydantic import BaseModel
from typing import List, Dict

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_predictions: List[Dict[str, float]]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_accuracy: float

class ClassListResponse(BaseModel):
    classes: List[str]