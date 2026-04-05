from pydantic import BaseModel
from typing import List, Dict, Optional

class Observation(BaseModel):
    compliance_score: float  # 0.0 to 1.0
    active_subscriptions: Dict[str, float] # {"Slack": 15.0, "Zoom": 20.0}
    social_sentiment: str # "Positive", "Negative", "Neutral"
    pending_tasks: int

class Action(BaseModel):
    category: str  # "LEGAL", "FINANCE", "SOCIAL"
    command: str   # "FIX_POLICY", "CANCEL_SUBSCRIPTION", "POST_REPLY"
    target_id: str # e.g., "Slack" or "Tweet_01"

class Reward(BaseModel):
    value: float
    reason: str