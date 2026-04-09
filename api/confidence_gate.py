THRESHOLD = 0.60

def evaluate_gate(sev_conf: float, pri_conf: float, queue_conf: float) -> dict:
    all_clear = all(c >= THRESHOLD for c in [sev_conf, pri_conf, queue_conf])
    overall_conf = (sev_conf + pri_conf + queue_conf) / 3.0
    
    low_fields = []
    if sev_conf < THRESHOLD:
        low_fields.append("severity")
    if pri_conf < THRESHOLD:
        low_fields.append("priority")
    if queue_conf < THRESHOLD:
        low_fields.append("assigned_to")
        
    return {
        "auto_assign": all_clear,
        "flagged_for_review": not all_clear,
        "low_confidence_fields": low_fields,
        "overall_confidence": overall_conf
    }
