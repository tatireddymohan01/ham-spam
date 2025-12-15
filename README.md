# 📱 Ham-Spam Classifier API

A production-ready REST API for SMS spam detection using Machine Learning (Logistic Regression + TF-IDF). Built with FastAPI and deployed on Azure App Service.

[![Deploy to Azure](https://img.shields.io/badge/Deploy%20to-Azure-0078D4?logo=microsoft-azure)](https://ham-spam-app-dnf0ghancgbchfcd.centralindia-01.azurewebsites.net)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)

## 🚀 Features

- **FastAPI Backend** - High-performance async REST API
- **ML-Powered** - Logistic Regression classifier with TF-IDF vectorization
- **Real-time Predictions** - Instant spam/ham classification with probability scores
- **Interactive UI** - Simple web interface for testing
- **Auto-documented** - OpenAPI/Swagger docs included
- **Docker Ready** - Containerized for easy deployment
- **CI/CD Pipeline** - Automated deployment to Azure via GitHub Actions
- **CORS Enabled** - Ready for frontend integration


## 🏗️ Project Structure

```
ham-spam/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application & routes
│   ├── model.py             # ML model loading & prediction logic
│   ├── config.py            # Configuration settings
│   ├── schemas.py           # Pydantic request/response models
│   ├── utils.py             # Helper functions
│   └── ham_spam_classifier.py
├── models/
│   └── spam_classifier.joblib   # Trained ML model
├── data/
│   └── SMSSpamCollection        # Training dataset
├── web/
│   └── index.html               # Web UI for testing
├── postman/
│   └── ham-spam-api.postman_collection.json
├── .github/
│   └── workflows/
│       └── azure-deploy.yml     # CI/CD pipeline
├── requirements.txt
├── Dockerfile
├── startup.sh                   # Azure startup script
└── README.md
```

## 📋 Prerequisites

- Python 3.10+
- pip or conda
- Docker (optional)
- Azure account (for deployment)

## 🔧 Local Development

### 1. Clone the Repository

```bash
git clone https://github.com/tatireddymohan01/ham-spam.git
cd ham-spam
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access the Application

- **API Documentation**: http://localhost:8000/docs
- **Web UI**: http://localhost:8000/web
- **API Root**: http://localhost:8000

## 🐳 Docker

```bash
docker build -t ham-spam-api .
docker run -p 8000:8000 ham-spam-api
```

Access at: http://localhost:8000

##  API Endpoints

### `GET /`
Health check endpoint

**Response:**
```json
{
  "message": "Ham-Spam Classifier API is running!",
  "ui": "/web"
}
```

### `POST /predict`
Classify a text message as spam or ham

**Request Body:**
```json
{
  "text": "Congratulations! You have won a $1000 gift card. Click here to claim."
}
```

**Response:**
```json
{
  "label": "spam",
  "probability_spam": 0.95,
  "probability_ham": 0.05
}
```

### `GET /web`
Serves the interactive web UI for testing

##  Azure Deployment

### Automated Deployment (GitHub Actions)

1. **Configure Azure Web App**
   - Create a Python 3.10 Web App on Azure
   - Download the publish profile
   - Set startup command: `bash startup.sh`

2. **Set GitHub Secret**
   - Go to repository Settings  Secrets  Actions
   - Add secret: `AZURE_WEBAPP_PUBLISH_PROFILE`
   - Paste the entire publish profile XML

3. **Deploy**
   - Push to `main` branch
   - GitHub Actions will automatically deploy

### Manual Deployment

```bash
# Login to Azure
az login

# Deploy using zip
az webapp deployment source config-zip \
  --resource-group <your-resource-group> \
  --name ham-spam-app \
  --src deploy.zip
```

##  Testing with Postman

Import the collection from `postman/ham-spam-api.postman_collection.json` into Postman to test all endpoints.

##  Model Details

- **Algorithm**: Logistic Regression
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Training Data**: SMS Spam Collection dataset
- **Output**: Binary classification (spam/ham) with probability scores

##  Technology Stack

- **Framework**: FastAPI
- **ML Libraries**: scikit-learn, pandas, numpy
- **Model Serialization**: joblib
- **Server**: Uvicorn (ASGI)
- **Validation**: Pydantic
- **Containerization**: Docker
- **Cloud**: Azure App Service
- **CI/CD**: GitHub Actions

##  Configuration

Edit `app/config.py` to modify:
- Model path
- Application name
- Version

##  Environment Variables

For Azure deployment, configure these in App Service settings:
- `PORT`: Application port (default: 8000)
- `WEBSITES_PORT`: 8000
- `SCM_DO_BUILD_DURING_DEPLOYMENT`: true

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m ''Add amazing feature''`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

##  License

This project is open source and available under the MIT License.

##  Author

**Mohan Tati Reddy**
- GitHub: [@tatireddymohan01](https://github.com/tatireddymohan01)

##  Acknowledgments

- SMS Spam Collection dataset
- FastAPI framework
- Azure App Service

---

**Live Demo**: [https://ham-spam-app-dnf0ghancgbchfcd.centralindia-01.azurewebsites.net](https://ham-spam-app-dnf0ghancgbchfcd.centralindia-01.azurewebsites.net)
