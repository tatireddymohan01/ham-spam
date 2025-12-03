# 📡 Ham-Spam Classifier API (FastAPI)

A production-level API for spam message detection using a Logistic Regression + TF-IDF model.

## 🚀 Features
- FastAPI backend with async endpoints
- Auto-loaded classifier model (joblib)
- Probabilistic spam detection
- Swagger docs included
- Docker support
- Ready for deployment on Render, Railway, AWS, Azure

## 🧠 Project Structure

```bash
ham-spam-api/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   ├── config.py
│   ├── schemas.py
│   └── utils.py
├── models/
│   └── spam_classifier.joblib   # <-- place your trained model here
├── web/
│   └── index.html               # simple HTML UI
├── postman/
│   └── ham-spam-api.postman_collection.json
├── requirements.txt
├── Dockerfile
└── README.md
```

## ▶️ Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open:
- API Docs: http://localhost:8000/docs
- Simple UI: open web/index.html in browser and point it to http://localhost:8000
```

## 🐳 Docker

```bash
docker build -t ham-spam-api .
docker run -p 8000:8000 ham-spam-api
```
