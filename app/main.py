from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.schemas import TextRequest, PredictionResponse
from app.model import SpamModel
from app.config import APP_NAME, VERSION

app = FastAPI(title=APP_NAME, version=VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SpamModel()

# Mount static files
web_dir = Path(__file__).parent.parent / "web"
app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

@app.get("/")
def home():
    return {"message": "Ham-Spam Classifier API is running!", "ui": "/web"}

@app.get("/web")
def web_ui():
    """Serve the web UI"""
    return FileResponse(str(web_dir / "index.html"))

@app.post("/predict", response_model=PredictionResponse)
def predict(req: TextRequest):
    label, prob_spam, prob_ham = model.predict(req.text)
    return PredictionResponse(
        label=label,
        probability_spam=prob_spam,
        probability_ham=prob_ham
    )
