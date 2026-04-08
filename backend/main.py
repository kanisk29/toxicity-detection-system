from fastapi import FastAPI
from backend.routes import router
from fastapi.middleware.cors import CORSMiddleware
from core.logging import get_logger

logger = get_logger(__name__)
logger.info("App started")
app = FastAPI()


app.include_router(router)
@app.get("/")
def home():
    return {"message": "API is running"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://127.0.0.1:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)