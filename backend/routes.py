from fastapi import APIRouter, Request
from pydantic import BaseModel
from backend.model_loader import predict_text
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

class TextRequest(BaseModel):
    text: str

@router.post("/predict")
@limiter.limit("60/minute") 
def predict(request: Request, req: TextRequest):
    results = predict_text(req.text)
    return {"results": results}