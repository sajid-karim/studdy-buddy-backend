# backend/app/models/test_models.py
from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Union
from enum import Enum

class TestResponse(BaseModel):
    id: str
    content: str
    config: dict

class QuestionType(str, Enum):
    MCQ = "mcq"
    CRQ = "crq"
    ERQ = "erq"

class TestConfig(BaseModel):
    subject: str
    grade: Union[str, int]  # Allow both string and int
    complexity: str
    questionType: QuestionType
    timeLimit: Union[str, int]  # Allow both string and int

    @field_validator('grade', 'timeLimit')
    def convert_to_string(cls, v):
        return str(v)

class Question(BaseModel):
    id: str
    question: str
    type: QuestionType
    topic: str
    points: int
    options: Optional[List[str]] = None
    correct_answer: Optional[str] = None
    rubric: Optional[List[str]] = None

class Test(BaseModel):
    id: str
    config: TestConfig
    questions: List[Question]
    content: str

class TestSubmission(BaseModel):
    answers: Dict[str, str]

    @field_validator('answers')
    def validate_answers(cls, v):
        # Convert any non-string values to strings
        return {
            str(k): str(v) if v is not None else ''
            for k, v in v.items()
        }

class SWOTAnalysis(BaseModel):
    strengths: List[str]
    weaknesses: List[str]
    opportunities: List[str]
    threats: List[str]
    topic_performance: Dict[str, Dict[str, float]]
    overall_score: float

class TestResult(BaseModel):
    test_id: str
    score: float
    analysis: SWOTAnalysis
    feedback: str