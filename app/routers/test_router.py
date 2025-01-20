# app/routers/test_router.py
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
import logging
import os
from typing import Dict, Any
from sqlalchemy.orm import Session
from ..schemas.test_schemas import TestConfig, TestSubmission, TestResult
from ..services.test_service import TestService
from ..database.database import get_db
from ..models.tests_models import DBTest, DBTestResult
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tests")

@router.post("/generate")
async def generate_tests_with_configurations(
    config: TestConfig,
    db: Session = Depends(get_db)
):
    """Generate a new test based on provided configuration."""
    try:
        logger.info(f"Received test configuration: {config.model_dump()}")
        
        # Validate OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured"
            )
        
        # Create TestService instance with database session
        test_service = TestService(db)
        
        # Generate test content using LLM
        test_content = await test_service.generate_test_with_llm(config)
        
        # Get the last created test from the database
        last_test = db.query(DBTest).order_by(DBTest.id.desc()).first()
        
        response_data = {
            "id": last_test.id,
            "content": test_content,
            "config": config.model_dump()
        }
        
        logger.info(f"Successfully generated test with ID: {response_data['id']}")
        return response_data
        
    except Exception as e:
        logger.error(f"Error in generate_tests_with_configurations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate test: {str(e)}"
        )

@router.post("/{test_id}/submit", response_model=TestResult)
async def submit_test(
    test_id: str,
    submission: TestSubmission,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> TestResult:
    """Submit and evaluate a completed test, generating comprehensive analysis."""
    try:
        logger.info(f"Received test submission for test ID: {test_id}")
        
        test_service = TestService(db)
        
        # Validate test existence
        test = test_service.get_test(test_id)
        if not test:
            raise HTTPException(status_code=404, detail="Test not found")
        
        # Evaluate test and generate analysis
        result = await test_service.evaluate_test(test_id, submission)
        
        logger.info(f"Completed test evaluation for test ID: {test_id}")
        
        return result
    
    except ValueError as ve:
        logger.error(f"Validation error in test submission: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error processing test submission: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process test submission")

@router.get("/{test_id}/analysis")
async def get_test_analysis(
    test_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Retrieve detailed analysis for a completed test."""
    try:
        logger.info(f"Retrieving analysis for test ID: {test_id}")
        
        test_service = TestService(db)
        
        # Get the test result from database
        result = test_service.get_test_result(test_id)
        if not result:
            raise HTTPException(
                status_code=404,
                detail="Analysis not found. Please submit the test first."
            )
        
        return {
            "test_id": result.test_id,
            "score": result.score,
            "analysis": result.analysis.model_dump(),
            "feedback": result.feedback
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving test analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve test analysis")

@router.post("/{test_id}/feedback")
async def generate_detailed_feedback(
    test_id: str,
    topic: str = None,
    include_recommendations: bool = True,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Generate detailed feedback for specific topics or the entire test."""
    try:
        logger.info(f"Generating detailed feedback for test ID: {test_id}")
        
        test_service = TestService(db)
        
        # Get test result from database
        result = test_service.get_test_result(test_id)
        if not result:
            raise HTTPException(
                status_code=404,
                detail="Analysis not found. Please submit the test first."
            )
        
        # Get analysis from the result
        analysis = result.analysis
        
        # Filter topic performance if topic specified
        if topic:
            topic_performance = {
                k: v for k, v in analysis.topic_performance.items()
                if k.lower() == topic.lower()
            }
        else:
            topic_performance = analysis.topic_performance
        
        feedback_data = {
            "topic_performance": topic_performance,
            "feedback": result.feedback
        }
        
        # Include recommendations if requested
        if include_recommendations:
            feedback_data["recommendations"] = analysis.opportunities
        
        return feedback_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating detailed feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate detailed feedback")

@router.post("/{test_id}/refresh-analysis")
async def refresh_test_analysis(
    test_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """Refresh the analysis for a test asynchronously."""
    try:
        logger.info(f"Queueing analysis refresh for test ID: {test_id}")
        
        test_service = TestService(db)
        
        # Validate test existence
        test = test_service.get_test(test_id)
        if not test:
            raise HTTPException(status_code=404, detail="Test not found")
        
        # Delete existing result if any
        db.query(DBTestResult).filter(DBTestResult.test_id == test_id).delete()
        db.commit()
        
        # Queue the analysis refresh
        # Note: This needs to be adjusted to work with the database session
        background_tasks.add_task(
            test_service.evaluate_test,
            test_id,
            None  # This should be replaced with the original submission if available
        )
        
        return {"status": "Analysis refresh queued successfully"}
        
    except Exception as e:
        logger.error(f"Error queueing analysis refresh: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to queue analysis refresh")