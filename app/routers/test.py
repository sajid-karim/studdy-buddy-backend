from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from ..database import get_db
from .. import schemas, crud

router = APIRouter(prefix="/api/tests", tags=["tests"])

@router.post("/generate")
async def generate_test(config: schemas.TestConfig, db: Session = Depends(get_db)):
    try:
        return await crud.create_test(db=db, config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{test_id}")
def get_test(test_id: int, db: Session = Depends(get_db)):
    test = crud.get_test(db, test_id)
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")
    return test
