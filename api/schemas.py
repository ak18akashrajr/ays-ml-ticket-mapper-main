from pydantic import BaseModel
from typing import List, Dict, Any

class TriageRequest(BaseModel):
    ticket_no: str
    created_at: str
    affected_user: str
    description: str

class TriageResponse(BaseModel):
    ticket_no: str
    severity: Dict[str, Any]
    priority: Dict[str, Any]
    assigned_to: Dict[str, Any]
    shap_reasons: List[str]
    auto_assign: bool
    flagged_for_review: bool
    low_confidence_fields: List[str]
    overall_confidence: float
    model_version: str
    inference_time_ms: int
