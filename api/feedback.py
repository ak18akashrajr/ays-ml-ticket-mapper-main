import json
import os
from datetime import datetime
from fastapi import APIRouter
from .schemas import FeedbackRequest

router = APIRouter()

FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'feedback.jsonl')

@router.post("/feedback")
def post_feedback(feedback: FeedbackRequest):
    record = feedback.dict()
    record["timestamp"] = datetime.utcnow().isoformat()
    
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
        
    return {"status": "success", "message": "Feedback recorded."}
