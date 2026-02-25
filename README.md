# 🤖 AI-Powered FAQ System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red?style=for-the-badge&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker)
![NLTK](https://img.shields.io/badge/NLTK-3.6-yellow?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24-orange?style=for-the-badge&logo=scikit-learn)

**An intelligent FAQ chatbot that automatically classifies user questions into 12 intent categories and returns the most relevant answer using Deep Learning.**

[🐳 Docker Hub](https://hub.docker.com/r/vikash4122002/faq-system) • [📖 API Docs](http://localhost:8000/docs) • [⭐ Give a Star](https://github.com/Vikash4122002/AI-POWERED-FAQ-SYSTEM)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [API Endpoints](#-api-endpoints)
- [Model Performance](#-model-performance)
- [Screenshots](#-screenshots)
- [Docker Deployment](#-docker-deployment)
- [Author](#-author)

---

## 🌟 Overview

The AI-Powered FAQ System is a production-ready intelligent chatbot built with **PyTorch ANN** for intent classification and **FastAPI** for the REST API backend.

It takes a user question, preprocesses it using NLP techniques, vectorizes it using TF-IDF, classifies the intent using a trained Neural Network, and retrieves the most relevant answer using cosine similarity — all in **under 1 millisecond**.

```
User Question → Preprocess → TF-IDF → ANN Model → Intent → Answer
```

---

## ✨ Features

- 🧠 **PyTorch Neural Network** — 3 hidden layers with BatchNorm and Dropout
- ⚡ **FastAPI REST API** — 7 endpoints with automatic Swagger docs
- 🎯 **12 Intent Categories** — password reset, billing, refund, security and more
- 💬 **Beautiful Chat UI** — dark themed frontend with confidence scores
- 🐳 **Docker Ready** — run entire project with 2 commands
- 📦 **Docker Hub Published** — publicly available image
- 🔄 **Batch Processing** — process up to 100 questions at once
- ⏱️ **< 1ms Response Time** — lightning fast predictions
- 📊 **90%+ Confidence** — high accuracy intent classification

---

## 🏗️ Architecture

```
                        AI-Powered FAQ System
                        ─────────────────────

User Question
     │
     ▼
┌─────────────────┐
│ TextPreprocessor│  → lowercase, remove stopwords,
│                 │    lemmatization, tokenization
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ TF-IDF          │  → converts text to 271
│ Vectorizer      │    numerical features
└─────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│           ANN Model                 │
│                                     │
│  Input(271)                         │
│      → Linear(256) + BatchNorm      │
│      → ReLU + Dropout(0.3)          │
│      → Linear(128) + BatchNorm      │
│      → ReLU + Dropout(0.3)          │
│      → Linear(64)  + BatchNorm      │
│      → ReLU + Dropout(0.3)          │
│      → Output(12 intents)           │
│                                     │
│  Total Parameters: 112,460          │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────┐
│ Cosine Similarity│  → finds best matching
│ Answer Retrieval │    answer from FAQ database
└─────────────────┘
     │
     ▼
   Answer ✅
```

---

## 💻 Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.10 |
| Deep Learning | PyTorch 2.1 |
| NLP | NLTK, TF-IDF |
| ML | scikit-learn |
| API Framework | FastAPI |
| Server | Uvicorn |
| Data | Pandas, NumPy |
| Containerization | Docker |
| Registry | Docker Hub |
| Frontend | HTML, CSS, JavaScript |

---

## 📁 Project Structure

```
AI-POWERED-FAQ-SYSTEM/
│
├── app/
│   ├── __init__.py
│   ├── inference.py      # Prediction pipeline
│   ├── main.py           # FastAPI app & endpoints
│   ├── schemas.py        # Pydantic request/response models
│   └── test_api.py       # API test script
│
├── data/
│   └── faq.csv           # 84 FAQ entries, 12 intents
│
├── ml/
│   ├── __init__.py
│   ├── preprocess.py     # Text cleaning & normalization
│   ├── vectorizer.py     # TF-IDF vectorizer
│   ├── model.py          # ANN classifier + trainer
│   ├── test_preprocess.py
│   └── test_vectorizer.py
│
├── saved_models/         # Generated after training
│   ├── faq_intent_model.pt
│   ├── vectorizer.pkl
│   └── intent_mappings.json
│
├── nginx/
│   └── nginx.conf        # Reverse proxy config
│
├── faq_chat.html         # Chat UI frontend
├── train.py              # Training script
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Option 1 — Docker (Recommended) 🐳

No Python setup needed! Just 2 commands:

```bash
docker pull vikash4122002/faq-system:latest
docker run -p 8000:8000 vikash4122002/faq-system:latest
```

Open: **http://localhost:8000/docs**

### Option 2 — Python Setup

```bash
# 1. Clone the repository
git clone https://github.com/Vikash4122002/AI-POWERED-FAQ-SYSTEM.git
cd AI-POWERED-FAQ-SYSTEM

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model
python train.py

# 5. Start the server
uvicorn app.main:app --reload
```

Open: **http://localhost:8000/docs**

### Option 3 — Docker Compose

```bash
docker-compose up --build
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | Service health check |
| POST | `/ask` | Ask a single question |
| POST | `/batch-ask` | Ask multiple questions (max 100) |
| GET | `/model-info` | Model architecture details |
| GET | `/intents` | List all 12 intent categories |
| GET | `/faq-stats` | FAQ knowledge base statistics |
| POST | `/feedback` | Submit prediction feedback |

### Example Request:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I reset my password?"}'
```

### Example Response:

```json
{
  "question": "How do I reset my password?",
  "answer": "Go to Settings > Security > Reset Password. You'll receive an email with instructions.",
  "intent": "password_reset",
  "confidence": 0.848,
  "processing_time_ms": 0.72
}
```

### Intent Categories:

```
✅ password_reset      ✅ account_delete
✅ refund_policy       ✅ contact_support
✅ pricing             ✅ billing
✅ cancellation        ✅ security
✅ features            ✅ account_management
✅ notifications       ✅ api
```

---

## 🧪 Testing

```bash
# Test preprocessor
python ml/test_preprocess.py

# Test vectorizer
python ml/test_vectorizer.py

# Test all API endpoints
python app/test_api.py
```

## 🐳 Docker Deployment

### Build locally:
```bash
docker build -t faq-system .
docker run -d -p 8000:8000 --name faq-app faq-system
```

### Pull from Docker Hub:
```bash
docker pull vikash4122002/faq-system:latest
docker run -d -p 8000:8000 --name faq-app vikash4122002/faq-system:latest
```

### Manage container:
```bash
docker stop faq-app      # Stop
docker start faq-app     # Start again
docker logs -f faq-app   # View logs
docker restart faq-app   # Restart
```

### Docker Hub:
```
hub.docker.com/r/vikash4122002/faq-system
Compressed Size: 4.12 GB
OS/ARCH: linux/amd64
```

---

## 🔧 Configuration

Key parameters in `train.py`:

```python
# Vectorizer
FAQVectorizer(
    max_features = 2000,
    ngram_range  = (1, 2),   # unigrams + bigrams
    use_idf      = True
)

# Model
IntentClassifier(
    hidden_dims  = [256, 128, 64],
    dropout_rate = 0.3,
    activation   = 'relu',
    use_batch_norm = True
)

# Training
trainer.train(
    epochs                  = 100,
    early_stopping_patience = 15,
    learning_rate           = 0.001
)
```

---

## 👨‍💻 Author

**Vikash Kumar**

- 📧 Email: vikash111107@gmail.com
- 🐙 GitHub: [github.com/Vikash4122002](https://github.com/Vikash4122002)
- 🐳 Docker Hub: [hub.docker.com/r/vikash4122002](https://hub.docker.com/r/vikash4122002)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

Made with ❤️ by Vikash Kumar

</div>
