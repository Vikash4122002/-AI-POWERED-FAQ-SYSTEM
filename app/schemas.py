"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any


class QuestionRequest(BaseModel):
    """Request model for the /ask endpoint"""
    question: str = Field(..., min_length=1, max_length=500,
                          description="The user's question about the product/service")

    @field_validator('question')
    @classmethod
    def question_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()


class AnswerResponse(BaseModel):
    """Response model for the /ask endpoint"""
    question: str = Field(..., description="Original question asked")
    answer: str = Field(..., description="The matched answer from FAQ")
    intent: str = Field(..., description="Predicted intent category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in ms")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "How do I reset my password?",
                "answer": "Go to Settings > Security > Reset Password.",
                "intent": "password_reset",
                "confidence": 0.97,
                "processing_time_ms": 45.2
            }
        }
    }


class HealthResponse(BaseModel):
    """Response model for /health endpoint"""
    status: str = Field(..., description="Service status: healthy / degraded")
    model_loaded: bool = Field(..., description="Whether the ANN model is loaded")
    vectorizer_loaded: bool = Field(..., description="Whether the vectorizer is loaded")
    total_faqs: Optional[int] = Field(None, description="Total FAQs in the knowledge base")
    device: Optional[str] = Field(None, description="Compute device (cpu / cuda)")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    status_code: int = Field(..., description="HTTP status code")


class BatchQuestionRequest(BaseModel):
    """Request model for /batch-ask endpoint"""
    questions: List[str] = Field(..., min_length=1, max_length=100,
                                 description="List of questions to process (max 100)")

    @field_validator('questions')
    @classmethod
    def questions_not_empty(cls, v):
        if not v:
            raise ValueError('Questions list cannot be empty')
        for i, q in enumerate(v):
            if not q or not q.strip():
                raise ValueError(f'Question at index {i} cannot be empty')
        return [q.strip() for q in v]


class BatchAnswerResponse(BaseModel):
    """Response model for /batch-ask endpoint"""
    results: List[AnswerResponse] = Field(..., description="List of answers")
    total_processed: int = Field(..., description="Total questions processed")
    total_time_ms: float = Field(..., description="Total processing time in ms")


class FeedbackRequest(BaseModel):
    """Request model for /feedback endpoint"""
    question: str = Field(..., description="Original question")
    predicted_intent: str = Field(..., description="Intent predicted by model")
    correct_intent: str = Field(..., description="Correct intent (from user)")
    confidence: float = Field(..., description="Model confidence score")
    comment: Optional[str] = Field(None, max_length=500, description="Optional comment")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "How do I reset my password?",
                "predicted_intent": "password_reset",
                "correct_intent": "password_reset",
                "confidence": 0.95,
                "comment": "Correct prediction"
            }
        }
    }


class ModelInfoResponse(BaseModel):
    """Response model for /model-info endpoint"""
    model_type: str
    input_dim: int
    hidden_layers: List[int]
    num_classes: int
    classes: List[str]
    vectorizer_params: Dict[str, Any]
    total_parameters: int
    device: str
