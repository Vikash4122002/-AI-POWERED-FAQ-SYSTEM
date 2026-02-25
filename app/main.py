"""
Main FastAPI application for the AI-Powered FAQ System
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging
from collections import Counter
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas import (
    QuestionRequest, AnswerResponse, HealthResponse, ErrorResponse,
    BatchQuestionRequest, BatchAnswerResponse, FeedbackRequest, ModelInfoResponse
)
from app.inference import FAQInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI-Powered FAQ System",
    description=(
        "An intelligent FAQ system powered by a PyTorch ANN classifier "
        "and TF-IDF vectorisation. Train the model with `python train.py`, "
        "then start the server with `uvicorn app.main:app --reload`."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Startup: load inference engine ───────────────────────────────────────────
START_TIME = time.time()

try:
    inference_engine = FAQInference(
        model_path='saved_models/faq_intent_model.pt',
        vectorizer_path='saved_models/vectorizer.pkl',
        faq_data_path='data/faq.csv',
        mappings_path='saved_models/intent_mappings.json'
    )
    logger.info("Inference engine loaded successfully")
except Exception as e:
    logger.error(f"Failed to load inference engine: {e}")
    logger.warning("Run `python train.py` to generate model artifacts before starting the server.")
    inference_engine = None


# ─── Middleware ────────────────────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{(time.time() - start) * 1000:.2f}ms"
    return response


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc), "status_code": 500}
    )


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to the AI-Powered FAQ System API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "ready" if inference_engine else "model_not_loaded",
        "endpoints": ["/health", "/ask", "/batch-ask", "/model-info", "/intents",
                      "/faq-stats", "/feedback"]
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    if inference_engine is None:
        return HealthResponse(
            status="degraded – run `python train.py` to generate model artifacts",
            model_loaded=False,
            vectorizer_loaded=False
        )
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        vectorizer_loaded=True,
        total_faqs=len(inference_engine.faq_questions),
        device=str(inference_engine.model.training)
    )


@app.post("/ask", response_model=AnswerResponse, tags=["Prediction"])
async def ask_question(request: QuestionRequest):
    """Ask a single question and receive the best-matching FAQ answer."""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run `python train.py` first.")

    try:
        result = inference_engine.get_answer(request.question)
        return AnswerResponse(
            question=result['question'],
            answer=result['answer'],
            intent=result['intent'],
            confidence=result['confidence'],
            processing_time_ms=result['processing_time_ms']
        )
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-ask", response_model=BatchAnswerResponse, tags=["Prediction"])
async def ask_batch(request: BatchQuestionRequest):
    """Process multiple questions in a single request (max 100)."""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run `python train.py` first.")

    start = time.time()
    try:
        results = inference_engine.get_batch_answers(request.questions)
        return BatchAnswerResponse(
            results=[
                AnswerResponse(
                    question=r['question'],
                    answer=r['answer'],
                    intent=r['intent'],
                    confidence=r['confidence'],
                    processing_time_ms=r.get('processing_time_ms')
                ) for r in results
            ],
            total_processed=len(results),
            total_time_ms=(time.time() - start) * 1000
        )
    except Exception as e:
        logger.error(f"Batch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Info"])
async def get_model_info():
    """Return metadata about the loaded model and vectorizer."""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return ModelInfoResponse(**inference_engine.get_model_info())


@app.get("/intents", tags=["Info"])
async def get_intents():
    """List all intent categories the model can classify."""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "intents": list(inference_engine.intent_to_idx.keys()),
        "count": len(inference_engine.intent_to_idx)
    }


@app.get("/faq-stats", tags=["Info"])
async def get_faq_stats():
    """Return statistics about the FAQ knowledge base."""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    intent_counts = Counter(inference_engine.faq_intents)
    return {
        "total_faqs": len(inference_engine.faq_questions),
        "total_intents": len(intent_counts),
        "intent_distribution": dict(intent_counts),
        "sample_questions": inference_engine.faq_questions[:5]
    }


@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(feedback: FeedbackRequest):
    """Submit correction feedback to help improve the model over time."""
    try:
        logger.info(f"Feedback received: {feedback.model_dump()}")
        # TODO: persist to a database (e.g. SQLite, Postgres)
        return {
            "message": "Feedback received – thank you!",
            "feedback_id": int(time.time())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
