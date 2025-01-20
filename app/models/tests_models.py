# app/models/tests_models.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from typing import Dict, List, Optional

Base = declarative_base()

class DBTest(Base):
    __tablename__ = "tests"

    id = Column(String, primary_key=True)
    subject = Column(String, nullable=False)
    grade = Column(Integer, nullable=False)
    complexity = Column(String, nullable=False)
    question_type = Column(String, nullable=False)
    time_limit = Column(Integer, nullable=False)
    content = Column(String, nullable=False)
    questions = relationship("DBQuestion", back_populates="test", cascade="all, delete-orphan")

class DBQuestion(Base):
    __tablename__ = "questions"

    id = Column(String, primary_key=True)
    test_id = Column(String, ForeignKey("tests.id"))
    question_text = Column(String, nullable=False)
    question_type = Column(String, nullable=False)
    topic = Column(String)
    points = Column(Integer, default=1)
    options = Column(JSON, nullable=True)  # For MCQ options
    correct_answer = Column(String, nullable=True)  # For MCQ answers
    rubric = Column(JSON, nullable=True)  # For CRQ/ERQ rubrics
    
    test = relationship("DBTest", back_populates="questions")

class DBTestResult(Base):
    __tablename__ = "test_results"

    id = Column(String, primary_key=True)
    test_id = Column(String, ForeignKey("tests.id"))
    score = Column(Float, nullable=False)
    topic_performance = Column(JSON)  # Store topic performance as JSON
    strengths = Column(JSON)  # Store as JSON array
    weaknesses = Column(JSON)  # Store as JSON array
    opportunities = Column(JSON)  # Store as JSON array
    threats = Column(JSON)  # Store as JSON array
    feedback = Column(String)

    test = relationship("DBTest")