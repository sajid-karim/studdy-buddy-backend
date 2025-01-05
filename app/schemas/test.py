from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class TestConfig(BaseModel):
    subject: str
    grade: int
    complexity: str
    question_type: str
    time_limit: int

class Question(BaseModel):
    question: str
    options: Optional[List[str]] = None
    correct_answer: str
    marks: int

class TestBase(BaseModel):
    subject: str
    grade: int
    complexity: str
    question_type: str
    time_limit: int
    questions: List[Question]

class Test(TestBase):
    id: int
    created_at: datetime
    
    class Config:
        orm_mode = True
