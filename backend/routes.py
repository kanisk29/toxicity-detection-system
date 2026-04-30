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
    output = predict_text(req.text)
    if isinstance(output,tuple):
        results,rewritten_text = output
    else:
        results,rewritten_text = output,None
    return {"results":results,"rewritten_text":rewritten_text}