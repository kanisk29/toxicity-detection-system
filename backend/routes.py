from fastapi import APIRouter
from pydantic import BaseModel
from backend.model_loader import predict_text

router = APIRouter()


class TextRequest(BaseModel):
    text: str


@router.post("/predict")
def predict(req: TextRequest):
    results = predict_text(req.text)

    return {
        "results": results
    }